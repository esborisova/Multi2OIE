import torch
from model import Multi2OIE

from typing import Union, List

from dataset import EvalDataset
from utils import utils
import torch
import time
from extract import simple_extract
from torch.utils.data import DataLoader


MODEL_DIR = "results/model-epoch1-step16000-score1.9768.bin"


class RelationModel:
    def __init__(self, device, batch_size=1):
        self.device = device
        self.batch_size = batch_size
        self.model = utils.simple_loader(MODEL_DIR, device)
        self.model.eval()

    def predict(self, text=Union[str, List[str]]):
        if isinstance(text, str):
            text = [text]

        text_loader = DataLoader(
            dataset=EvalDataset(
                text, max_len=64, tokenizer_config="bert-base-multilingual-cased"
            ),
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
        )
        triplets, confidence = simple_extract(self.model, text_loader, self.device)
        return (triplets, confidence)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = RelationModel(device, batch_size=3)

text = [
    "Lasse er en sød dreng og vandt guldmedaljen i 100 meter fri",
    "Pernille Blume vandt EM",
    "Hvem er det der banker, det er Peter Anker",
]

text = "Lasse er en sød dreng og vandt guldmedaljen i 100 meter fri. Pernille Blume vandt sølv til  EM. Hvem er det der banker, det er Peter Anker"


import spacy
from spacy.language import Language
from spacy.tokens import Doc


# add default batch size (and as argument)
@Language.factory("relation_extraction")
def create_relation_component(nlp, name):
    return RelationExtraction(nlp)


class RelationExtraction:
    def __init__(self, nlp):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = RelationModel(device, 64)
        if not Doc.has_extension("relations"):
            Doc.set_extension("relations", default=[])
        if not Doc.has_extension("confidence"):
            Doc.set_extension("rel_confidence", default=[])

    def __call__(self, doc):

        for sent in doc.sents:
            rels, conf = self.model.predict(sent.text)
            doc._.relations.append(rels)
            doc._.rel_confidence.append(conf)
        return doc


nlp = spacy.load("da_core_news_sm")
nlp.add_pipe("relation_extraction")
doc = nlp(text)
doc._.relations
doc._.rel_conf
