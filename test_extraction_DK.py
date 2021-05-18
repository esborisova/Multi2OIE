from dataset import EvalDataset
from utils import utils
import torch
import time
from extract import extract_to_list
from torch.utils.data import DataLoader

output_path = "results/epoch1_dev/step8000/carb_dev/"
dev_gold = "evaluate/OIE2016_dev.txt"

model_path = "results/model-epoch1-step16000-score1.9768.bin"


test_sents = [
    "Lasse er en dreng på 26 år.",
    "Jeg arbejder som tømrer",
    "Albert var videnskabsmand og døde i 1921",
    "Lasse lives in Denmark and owns two cats",
]

batch_size = 64

test_loader = DataLoader(
    dataset=EvalDataset(
        test_sents, max_len=64, tokenizer_config="bert-base-multilingual-cased"
    ),
    batch_size=batch_size,
    num_workers=4,
    pin_memory=True,
)

bert_config = "bert-base-multilingual-cased"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
binary = False

model = utils.get_models(
    bert_config=bert_config,
    pred_n_labels=3,
    arg_n_labels=9,
    n_arg_heads=8,
    n_arg_layers=4,
    pos_emb_dim=64,
    use_lstm=False,
    device=device,
)

model.load_state_dict(torch.load(model_path))
model.zero_grad()
model.eval()

args = {"bert_config": bert_config, "device": device, "binary": binary}

start = time.time()
extractions = extract_to_list(args, model, test_loader)
print("TIME: ", time.time() - start)
# test_results = do_eval(args.save_path, args.test_gold_path)
# utils.print_results("TEST RESULT", test_results, ["F1  ", "PREC", "REC ", "AUC "])
