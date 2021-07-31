import torch
from relation_model import RelationModel
from spacy.language import Language
from spacy.tokens import Doc
from download_model import DEFAULT_CACHE_DIR
import numpy as np


@Language.factory(
    "relation_extraction", default_config={"batch_size": 64, "path": DEFAULT_CACHE_DIR}
)
def create_relation_component(nlp: Language, name: str, batch_size: int, path: str):
    return RelationExtraction(nlp, batch_size, path)


class RelationExtraction:
    """Adds relations triplets for each sentence to a Doc."""

    def __init__(self, nlp: Language, batch_size: int, path: str):
        self.batch_size = batch_size
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = RelationModel(device, batch_size, path)

        Doc.set_extension("relations", default=[], force=True)
        Doc.set_extension("rel_confidence", default=[], force=True)

    def __call__(self, doc: Doc):

        text = [sent.text for sent in doc.sents]
        n_texts = len(text)
        if n_texts > self.batch_size:
            n_splits = int(np.ceil(n_texts / self.batch_size))
            rels = []
            conf = []
            for split in range(n_splits):
                idx = split * self.batch_size
                if split == n_splits - 1:
                    tmp_rels, tmp_conf = self.model.predict(text[idx:])
                else:
                    tmp_rels, tmp_conf = self.model.predict(
                        text[idx : idx + self.batch_size]
                    )
                rels.extend(tmp_rels)
                conf.extend(tmp_conf)
        else:
            rels, conf = self.model.predict(text)
        doc._.relations = rels
        doc._.rel_confidence = conf
        return doc
