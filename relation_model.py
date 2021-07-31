"""Bundles model and adds predict method """
from typing import Union, List

from dataset import EvalDataset
from utils import utils
from extract import simple_extract
from torch.utils.data import DataLoader

from download_model import download_model, DEFAULT_CACHE_DIR


class RelationModel:
    def __init__(self, device, batch_size=1, path=DEFAULT_CACHE_DIR):
        self.device = device
        self.batch_size = batch_size
        path = download_model(save_path=path)
        self.model = utils.simple_loader(path, device)
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
