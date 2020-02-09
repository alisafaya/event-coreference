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


def predict_set(infile, outfile):
    docs = []
    for p, n in get_instance_pairs(infile):
        instance_pairs = []
        label = 1
        for i, (x, y) in enumerate(p + n):
            if i == len(p):
                label = 0
            _x, _y, _ = prepare_pair(x, y, -1)
            _x, _y = get_representation(_x), get_representation(_y)
            prediction = (1 if scorer_model(_x, _y).item() >= 0.5 else 0)
            pair =  {"sentence_pair": (" ".join(x["tokens"]), " ".join(y["tokens"])), "label": label, "prediction": prediction }
            instance_pairs.append(pair)
        docs.append(instance_pairs)

    with open(outfile, "w") as fo:
        fo.write(json.dumps(docs, indent=2, ensure_ascii=False))            


if __name__ == "__main__":
    print("Loading data ...")
    positives, negatives = [], []
    for p, n in get_instance_pairs("event_trigger_sentence_for_coreference_no_overlap.json"):
        positives.append(p)
        negatives.append(n)
    
    print("Loaded", len(positives), "documents.")
    positives, negatives = np.array(positives), np.array(negatives)
    indices = np.arange(len(positives)) 
    np.random.shuffle(indices)
    positives, negatives = positives[indices], negatives[indices]

    # debug
    # positives, negatives = positives, negatives

    doc_no = len(positives)
    p_train, p_dev, p_test = positives[:int(doc_no*0.70)], positives[int(doc_no*0.70):int(doc_no*0.95)], positives[int(doc_no*0.95):]
    n_train, n_dev, n_test = negatives[:int(doc_no*0.70)], negatives[int(doc_no*0.70):int(doc_no*0.95)], negatives[int(doc_no*0.95):]

    train = prepare_set(p_train, n_train)
    dev = prepare_set(p_dev, n_dev)
    test = prepare_set(p_test, n_test)

    scorer_model = Scorer()
    scorer_model = scorer_model.to(device)

    epochs = 10
    bestloss = np.Inf
    bestprecision = 0
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(scorer_model.parameters(), lr=0.00005)

    scorer_model.zero_grad()
    for epoch in range(1, epochs+1):
        print("Epoch", epoch)
        total_loss = 0
        random.shuffle(train)
        for i, (x, y, label) in enumerate(train):

            outputs = scorer_model(x, y)
            loss = criterion(outputs, label)
            total_loss += loss.item()
            loss.backward()
    
            if i % 32 == 0:
                optimizer.step()
                optimizer.zero_grad()

        optimizer.step()
        optimizer.zero_grad()

        print("Train loss:", total_loss)        

        # calculate dev loss
        dev_loss = 0
        y_pred, y_dev = [], [] 
        for x, y, label in dev:
            outputs = scorer_model(x, y)
            y_pred.append(1 if outputs.item() >= 0.5 else 0)
            y_dev.append(label.item())
            loss = criterion(outputs, label)
            dev_loss += loss.item()

        print("Dev loss:", dev_loss)      
        # if bestloss > dev_loss:
        #     bestloss = dev_loss
        #     torch.save(scorer_model.state_dict(), "scorer_model.pt")

        precision = precision_score(y_dev, y_pred)
        print("Dev precision:", precision)  
        if precision > bestprecision:
            bestprecision = precision
            torch.save(scorer_model.state_dict(), "scorer_model.pt")

    scorer_model.load_state_dict(torch.load("scorer_model.pt"))
    scorer_model.eval()

    y_pred, y_train = [], [] 
    for x, y, label in train: 
        y_pred.append(1 if scorer_model(x, y).item() >= 0.5 else 0) 
        y_train.append(label.item())
    print("Train :", classification_report(y_train, y_pred))
    print(confusion_matrix(y_train, y_pred))

    y_pred, y_dev = [], [] 
    for x, y, label in dev: 
        y_pred.append(1 if scorer_model(x, y).item() >= 0.5 else 0) 
        y_dev.append(label.item())
    print("Dev :", classification_report(y_dev, y_pred))
    print(confusion_matrix(y_dev, y_pred))

    y_pred, y_test = [], [] 
    for x, y, label in test: 
        y_pred.append(1 if scorer_model(x, y).item() >= 0.5 else 0) 
        y_test.append(label.item())
    print("Test :", classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    y_pred, y_true = [], [] 
    for x, y, label in (test + dev + train):
        y_pred.append(1) 
        y_true.append(label.item())
    print("Baseline results :", classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))









    

