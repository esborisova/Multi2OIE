from dataset import EvalDataset
from utils import utils
import torch
import time
from extract import extract_to_list
from torch.utils.data import DataLoader
from typing import Optional, List, Tuple
DEFAULT_MODEL_DIR = Path(Path.home(), ".relation_model")
class RelationExtraction(): 

    def __init__(self, model_path: Optional[str] = None, batch_size: int = 64, max_len: int = 64, num_workers: int = 4, pin_memory: bool = True):
        self._bert_config  = "bert-base-multilingual-cased"
        self._binary = False
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._batch_size = batch_size
        self._max_len= max_len
        self._num_workers= num_workers
        self._args = {"bert_config": self._bert_config, "device": self._device, "binary": self._binary}
        self._bert_model = self.prepare_model(model_path)
        self._pin_memory = pin_memory
    

# Function for loading a model

    def load_model(self, path):
        model = utils.get_models(
            bert_config = self._bert_config,
            pred_n_labels=3,
            arg_n_labels=9,
            n_arg_heads=8,
            n_arg_layers=4,
            pos_emb_dim=64,
            use_lstm=False,
            device = self._device)

        model.load_state_dict(torch.load(path))
        model.zero_grad()
        model.eval()

        return model

# Function for preparing a model. If the path is given, it loads the model. 
# Otherwise, it creates a folder in the home directory. Then it dowloads the model, saves it to the specified location and loads it.

    def prepare_model(self, model_path): 
        if model_path is None:

            import urllib.request
            import os 
            from pathlib import Path
            import earthpy as et
            
            new_path = os.path.join(et.io.HOME, "Multi2OIE_model")
            if (not Path(new_path).exists()):
                os.mkdir(new_path)

            model_path = os.path.join(new_path, "model.bin")
            urllib.request.urlretrieve(
                url='https://sciencedata.dk//shared/81ee2688645634814152e3965e74b7f7?download', 
                filename =  model_path
                )

        return self.load_model(path = model_path)

    # Function for preparing the dataset

    def prepare_data(self, sents):
        dataset=EvalDataset(sents, self._max_len, self._bert_config)
        test_loader = DataLoader( 
            dataset,
            self._batch_size,
            self._num_workers,
            pin_memory=self._pin_memory
            )
        return test_loader


    # Function for extracting relations from a given dataset
 
    def extract_relations(self, text: List[str], verbose: bool = False) -> List[Tuple]:
        if verbose:
            start = time.time()
        prepared_sent = self.prepare_data(sents = text)
        extractions = extract_to_list(self._args, self._bert_model, prepared_sent)
        if verbose:
            print("TIME: ", time.time() - start)
        return extractions


# Testing 

test_sents = [
    "Lasse er en dreng på 26 år.",
    "Jeg arbejder som tømrer",
    "Albert var videnskabsmand og døde i 1921",
    "Lasse lives in Denmark and owns two cats",
]


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


relations = RelationExtraction()

final_result = relations.extract_relations(test_sents)

print(final_result ["sentence"])
print(final_result ["extraction_3"])


