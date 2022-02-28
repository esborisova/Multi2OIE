from typing import Callable, Dict, Iterable, Iterator, List

import spacy
from spacy import Vocab
from spacy.language import Language
from spacy.ml import CharacterEmbed
from spacy.pipeline import TrainablePipe
from spacy.tokens import Doc
from spacy.util import minibatch
from spacy_transformers import FullTransformerBatch
from spacy_transformers.annotation_setters import null_annotation_setter
from spacy_transformers.util import batch_by_length

from spacy_transformers.align import get_alignment
from thinc.api import Model, PyTorchWrapper, chain, with_array
from transformers import AutoTokenizer

from relationextraction import KnowledgeTriplets
from relationextraction.model import Multi2OIE
from relationextraction.util import split_by_doc

from relationextraction.util import wp2tokid, wp_span_to_token, install_extension

# TODO:
# Rewrite documentation
# Rename things
# Remove fluff
# Handle empty strings


@Language.factory(
    "relation_extractor",
    default_config={
        "max_batch_items": 50,
        "confidence_threshold": 1.0,
        "labels": [],
        "model_args": {
            "batch_size": 64,
        },
    },
)
def make_relation_extractor(
    nlp: Language,
    name: str,
    confidence_threshold: float,
    max_batch_items: int,
    labels: List,
    model_args: Dict,
):
    return SpacyRelationExtractor(
        nlp.vocab,
        name=name,
        confidence_threshold=confidence_threshold,
        labels=labels,
        max_batch_items=max_batch_items,
        model_args=model_args,
    )


class SpacyRelationExtractor(TrainablePipe):
    """spaCy pipeline component that provides access to a transformer model from
    the Huggingface transformers library. Usually you will connect subsequent
    components to the shared transformer using the TransformerListener layer.
    This works similarly to spaCy's Tok2Vec component and Tok2VecListener
    sublayer.
    The activations from the transformer are saved in the doc._.trf_data extension
    attribute. You can also provide a callback to set additional annotations.
    Args:
        vocab (Vocab): The Vocab object for the pipeline.
        model (Model[List[Doc], FullTransformerBatch]): A thinc Model object wrapping
            the transformer. Usually you will want to use the TransformerModel
            layer for this.
        set_extra_annotations (Callable[[List[Doc], FullTransformerBatch], None]): A
            callback to set additional information onto the batch of `Doc` objects.
            The doc._.{doc_extension_trf_data} attribute is set prior to calling the callback
            as well as doc._.{doc_extension_prediction} and doc._.{doc_extension_prediction}_prob.
            By default, no additional annotations are set.
        labels (List[str]): A list of labels which the transformer model outputs, should be ordered.
    """

    def __init__(
        self,
        vocab: Vocab,
        name: str,
        labels: List[str],
        confidence_threshold: float,
        max_batch_items: int,  # Max size of padded batch
        model_args,
    ):
        """Initialize the transformer component."""
        self.vocab = vocab
        self.model = KnowledgeTriplets(**model_args)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.confidence_threshold = confidence_threshold
        # if not isinstance(self.model, Model):
        #     raise ValueError(f"Expected Thinc Model, got: {type(self.model)}")
        self.cfg = {"max_batch_items": max_batch_items}

        [
            install_extension(ext)
            for ext in [
                "relation_triplets",
                "relation_head",
                "relation_relation",
                "relation_tail",
                "relation_confidence",
            ]
        ]

    def set_annotations(self, docs: Iterable[Doc], predictions: Dict) -> None:
        """Assign the extracted features to the Doc objects. By default, the
        TransformerData object is written to the doc._.{doc_extension_trf_data} attribute. Your
        set_extra_annotations callback is then called, if provided.
        Args:
            docs (Iterable[Doc]): The documents to modify.
            predictions: (FullTransformerBatch): A batch of activations.
        """
        # get nested list of indices above confidence threshold
        filtered_indices = [
            [
                idx
                for idx, conf in enumerate(confidences)
                if conf > self.confidence_threshold
            ]
            for confidences in (predictions["confidence"])
        ]
        # only keep relations above the threshold
        for key in ["extraction", "extraction_span", "confidence"]:
            predictions[key] = [
                [values[filter_idx] for filter_idx in indices]
                for indices, values in zip(filtered_indices, predictions[key])
            ]

        # Calculating alignment between wordpieces and spacy tokenizer
        align = [
            get_alignment([doc], [wp], self.tokenizer.all_special_tokens)
            for doc, wp in zip(docs, predictions["wordpieces"])
        ]
        # getting wordpiece to token id mapping
        wp2tokids = [wp2tokid(aligned_doc) for aligned_doc in align]
        # transforming wp span to spacy Span
        extraction_spans = [
            wp_span_to_token(relation_span, wp2tokid_mapping, spacy_doc)
            for relation_span, wp2tokid_mapping, spacy_doc in zip(
                predictions["extraction_span"], wp2tokids, docs
            )
        ]
        for idx, (doc, data) in enumerate(zip(docs, extraction_spans)):

            setattr(doc._, "relation_triplets", data["triplet"])
            setattr(
                doc._,
                "relation_confidence",
                predictions["confidence"][idx],
            )
            setattr(doc._, "relation_head", data["head"])
            setattr(doc._, "relation_relation", data["relation"])
            setattr(doc._, "relation_tail", data["tail"])

    def __call__(self, doc: Doc) -> Doc:
        """Apply the pipe to one document. The document is modified in place,
        and returned. This usually happens under the hood when the nlp object
        is called on a text and all components are applied to the Doc.
        docs (Doc): The Doc to process.
        RETURNS (Doc): The processed Doc.
        DOCS: https://spacy.io/api/transformer#call
        """
        outputs = self.predict([doc])
        self.set_annotations([doc], outputs)
        return doc

    def pipe(self, stream: Iterable[Doc], *, batch_size: int = 128) -> Iterator[Doc]:
        """Apply the pipe to a stream of documents. This usually happens under
        the hood when the nlp object is called on a text and all components are
        applied to the Doc.
        stream (Iterable[Doc]): A stream of documents.
        batch_size (int): The number of documents to buffer.
        YIELDS (Doc): Processed documents in order.
        DOCS: https://spacy.io/api/transformer#pipe
        """
        for outer_batch in minibatch(stream, batch_size):
            outer_batch = list(outer_batch)
            outer_batch_text = [doc.text for doc in outer_batch]
            self.set_annotations(outer_batch, self.predict(outer_batch_text))
            # for indices in batch_by_length(outer_batch, self.cfg["max_batch_items"]):
            #     subbatch = [outer_batch[i] for i in indices]
            #     subbatch_texts = [subbatch_doc.text for subbatch_doc in subbatch]
            #    self.set_annotations(subbatch, self.predict(subbatch_texts))
            yield from outer_batch

    def predict(self, docs: Iterable[Doc]) -> FullTransformerBatch:
        """Apply the pipeline's model to a batch of docs, without modifying them.
        Returns the extracted features as the FullTransformerBatch dataclass.
        docs (Iterable[Doc]): The documents to predict.
        RETURNS (FullTransformerBatch): The extracted features.
        DOCS: https://spacy.io/api/transformer#predict
        """
        if not any(len(doc) for doc in docs):
            # Handle cases where there are no tokens in any docs.
            activations = FullTransformerBatch.empty(len(docs))
        else:
            activations = self.model.extract_relations(docs)
        return activations


if __name__ == "__main__":
    nlp = spacy.load("da_core_news_sm")

    test_sents = [
        "Pernille Blume vinder delt EM-sølv i Ungarn.",
        "Pernille Blume blev nummer to ved EM på langbane i disciplinen 50 meter fri.",
        "Hurtigst var til gengæld hollænderen Ranomi Kromowidjojo, der sikrede sig guldet i tiden 23,97 sekunder.",
        "Og at formen er til en EM-sølvmedalje tegner godt, siger Pernille Blume med tanke på, at hun få uger siden var smittet med corona.",
        "Ved EM tirsdag blev det ikke til medalje for den danske medley for mixede hold i 4 x 200 meter fri.",
        "In a phone call on Monday, Mr. Biden warned Mr. Netanyahu that he could fend off criticism of the Gaza strikes for only so long, according to two people familiar with the call",
        "That phone call and others since the fighting started last week reflect Mr. Biden and Mr. Netanyahu’s complicated 40-year relationship.",
        "Politiet skal etterforske Siv Jensen etter mulig smittevernsbrudd.",
        "En av Belgiens mest framträdande virusexperter har flyttats med sin familj till skyddat boende efter hot från en beväpnad högerextremist.",
    ]

    config = {
        "max_batch_items": 15,
        "confidence_threshold": 1.0,
    }

    # model = KnowledgeTriplets()

    nlp.add_pipe("relation_extractor", config=config)

    pipe = nlp.pipe(test_sents)
    for d in pipe:
        print(d.text, "\n", d._.relation_triplets)
