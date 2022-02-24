from relationextraction import KnowledgeTriplets

import spacy
from spacy.tokens import Doc, Span

from spacy_transformers.align import get_alignment

from transformers import AutoTokenizer
from thinc.types import Ragged

from typing import List, Dict, Optional, Tuple


def wp2tokid(align: Ragged) -> Dict[int, int]:
    """Gets the wordpiece to token id mapping"""
    wp2tokid = []
    for i, length in enumerate(align.lengths):
        wp2tokid += length * [i]

    return {int(wp_id): t_id for wp_id, t_id in zip(align.data, wp2tokid)}


def wp_span_to_token(
    relation_span: List[List[int]], wp_tokenid_mapping: Dict, doc: Doc
) -> List[Dict[str, List[Span]]]:
    """
    Converts the wp span for each relation to spans.
    Assumes that relations are contiguous"""
    relations = {"triplet": [], "head": [], "relation": [], "tail": []}
    for triplet in relation_span:
        # convert list of wordpieces in the extraction to a tuple of the span (start, end)
        head = get_wp_span_tuple(triplet[0])
        relation = get_wp_span_tuple(triplet[1])
        tail = get_wp_span_tuple(triplet[2])

        # convert the wp span to token span
        head = wp_to_token_id_mapping(head, wp_tokenid_mapping)
        relation = wp_to_token_id_mapping(relation, wp_tokenid_mapping)
        tail = wp_to_token_id_mapping(tail, wp_tokenid_mapping)

        # convert token span to spacy span
        head = token_span_to_spacy_span(head, doc)
        relation = token_span_to_spacy_span(relation, doc)
        tail = token_span_to_spacy_span(tail, doc)

        relations["head"].append(head)
        relations["relation"].append(relation)
        relations["tail"].append(tail)
        relations["triplet"].append((head, relation, tail))
    return relations


def get_wp_span_tuple(span: List[int]) -> Tuple[int, int]:
    """Converts the relation span to a tuple, assumes that extractions are contiguous"""
    if not span:
        return ""
    if len(span) == 1:
        return (span[0], span[0])
    else:
        return (span[0], span[-1])


def wp_to_token_id_mapping(
    span: Tuple[int, int], wp_tokenid_mapping: Dict[int, int]
) -> Tuple[int, int]:
    """converts wordpiece spans to token ids"""
    if span:
        return (wp_tokenid_mapping[span[0]], wp_tokenid_mapping[span[1]])
    else:
        return ""


def token_span_to_spacy_span(span: Tuple[int, int], doc: Doc):
    """converts token id span to span"""
    if not span:
        return ""
    else:
        return doc[span[0] : span[1] + 1]


## TODO
# Wrap i en trainable pipe


class RelationComponent:
    def __init__(self, nlp: Language):
        self.model = KnowledgeTriplets()

        if not Doc.has_extension("relation_triplets"):
            Doc.set_extension("relation_triplets", default=[])


if __name__ == "__main__":
    rel_extractor = KnowledgeTriplets()

    text = [
        "Lasse er en sød fyr på 27 år",
        "Han bor i Aarhus og har en kæreste der hedder Solvej som også har en hund der hedder Jens",
    ]

    triplets = rel_extractor.extract_relations(text)

    nlp = spacy.load("da_core_news_lg")
    docs = nlp.pipe(triplets["sentence"])

    spacy_docs = [doc for doc in docs]
    wordpieces = triplets["wordpieces"]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    # have to make doc and wp a list to avoid errors..
    # doing it without list comprehension returns a flat list so we lose information on doc
    align = [
        get_alignment([doc], [wp], tokenizer.all_special_tokens)
        for doc, wp in zip(spacy_docs, wordpieces)
    ]

    relation_spans = triplets["extraction_span"]
    wp2tokids = [wp2tokid(aligned_doc) for aligned_doc in align]

    extraction_spans = [
        wp_span_to_token(relation_span, wp2tokid_mapping, spacy_doc)
        for relation_span, wp2tokid_mapping, spacy_doc in zip(
            triplets["extraction_span"], wp2tokids, spacy_docs
        )
    ]
