from transformers import *
from data import get_instance_pairs
import random
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_score
import json

# Tell pytorch to run this model on the GPU.
use_gpu = True
seed = 1234 # set none to ignore seeding
max_length = 256
hidden_layer = 4
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertModel.from_pretrained('albert-base-v2', output_hidden_states=True)

# If there's a GPU available...
device = "cpu"
if use_gpu and torch.cuda.is_available():
    device = torch.device("cuda")
    model.cuda()
else:
    device = torch.device("cpu")
model.eval()

if seed is not None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def prepare_pair(x, y, label, max_length=256):
    """returns input_ids, input_masks, labels for pair of sentences in BERT input format"""
    global tokenizer
    x = tokenizer.encode_plus(" ".join(x["tokens"]), pad_to_max_length=True, add_special_tokens=True, max_length=max_length)
    y = tokenizer.encode_plus(" ".join(y["tokens"]), pad_to_max_length=True, add_special_tokens=True, max_length=max_length)

    x = (torch.tensor(x["input_ids"]).unsqueeze(0), torch.tensor(x["attention_mask"]).unsqueeze(0), torch.tensor(x["token_type_ids"]).unsqueeze(0))
    y = (torch.tensor(y["input_ids"]).unsqueeze(0), torch.tensor(y["attention_mask"]).unsqueeze(0), torch.tensor(y["token_type_ids"]).unsqueeze(0))
    label = torch.tensor(label).float()
    return x, y, label


def get_representation(x):
    """returns sentence representation by pooling over hidden states of the model"""
    global model
    with torch.no_grad():
        x = tuple(i.to(device) for i in x)
        x_output = model(input_ids=x[0], attention_mask=x[1], token_type_ids=x[2])
        averaged_hidden_states = torch.stack(x_output[2][-hidden_layer:]).mean(0)
        pooled = averaged_hidden_states[:, :x[1].sum(), :].mean(1) 
    return pooled.clone().detach()


def prepare_set(p_set, n_set):
    pset = []
    for p, n in zip(p_set, n_set):
        instance_pairs = []
        pairs = [ prepare_pair(pair[0], pair[1], 1, max_length=max_length) for pair in p ] + [ prepare_pair(pair[0], pair[1], 0, max_length=max_length) for pair in n ]
        random.shuffle(pairs)
        for x, y, label in pairs:
            instance_pairs.append((get_representation(x), get_representation(y), label.to(device)))
        
        # pset.append(instance_pairs)
        pset += instance_pairs
    return pset


class Scorer(torch.nn.Module):
    """MLP model used for pairwise scoring"""
    def __init__(self, input_size=1536):
        super(Scorer, self).__init__()
        self.lin = torch.nn.Linear(input_size, input_size // 2, bias=True)
        self.lin2 = torch.nn.Linear(input_size // 2, 1, bias=True)
    
    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        x = self.lin(x)
        x = torch.tanh(x)
        x = self.lin2(x)
        return torch.sigmoid(x).squeeze()


def predict_set(infile, outfile, return_proba=False):
    docs = []
    for p, n in get_instance_pairs(infile):
        instance_pairs = []
        label = 1
        for i, (x, y) in enumerate(p + n):
            if i == len(p):
                label = 0
            _x, _y, _ = prepare_pair(x, y, -1)
            _x, _y = get_representation(_x), get_representation(_y)
            prediction = scorer_model(_x, _y).item() if return_proba else (1 if scorer_model(_x, _y).item() >= 0.5 else 0) 
            pair =  {"sentence_pair": (" ".join(x["tokens"]), " ".join(y["tokens"])), "label": label, "prediction": prediction }
            instance_pairs.append(pair)
        docs.append(instance_pairs)

    with open(outfile, "w") as fo:
        fo.write(json.dumps(docs, indent=2, ensure_ascii=False))            


if __name__ == "__main__":

    scorer_model = Scorer()
    scorer_model.load_state_dict(torch.load("scorer_model.pt"))
    scorer_model = scorer_model.to(device)
    scorer_model.eval()

    predict_set("pipeline_event_sentences.json", "pipeline_event_sentences.proba.predicted.json", return_proba=True)
    








    

