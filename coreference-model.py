from transformers import *
import random
import torch
import numpy as np
import json
import networkx as nx


class Solver:
    """
    An implementation of the disagreement-minimizing correlation clustering algorithm by Bansal, Blum and Chawla [1].
    Bansal, Nikhil, Avrim Blum, and Shuchi Chawla. “Correlation Clustering.” Machine Learning 56, no. 1–3 (2004): 89–113.
    https://github.com/filkry/py-correlation-clustering/blob/master/py_correlation_clustering.py
    """
    def __init__(self, G, delta = 1.0/44):
        """
        Args:
            delta: "cleanness" parameter. Defaults to the assumed value of 1/44
                   given in the paper
        """
        self.__G__ = G
        self.__reset_caches__()
        self.__clusters__ = None
        self.__delta__ = delta

    def __reset_caches__(self):
        self.__G_nodes__ = set(self.__G__.nodes())
        self.__N_plus_cache__ = dict()

    def __remove_cluster__(self, C):
        self.__G__.remove_nodes_from(C)
        self.__reset_caches__()

    def positive_neighbours(self, u):
        """
        Returns N+(u), or {u} U {v : e(u, v) = +}
        Args:
            G: a networkx graph where presence of edges indicates a + edge
            u: a node in G
        """

        if u in self.__N_plus_cache__:
            return self.__N_plus_cache__[u]

        res = set([u])
        for i in self.__G__.neighbors(u):
            res.add(i)

        self.__N_plus_cache__[u] = res
        return res

    def delta_good(self, v, C, delta):
        """
        Returns true if v is delta-good with respect to C, where C is a cluster in
        G
        Args:
            G: a networkx graph
            v: a vertex v in G
            C: a set of vertices in G
            delta: "cleanness" parameter
        """

        Nv = self.positive_neighbours(v)

        return (len(Nv & C) >= (1.0 - delta) * len(C) and
                len(Nv & (self.__G_nodes__ - C)) <= delta * len(C))

    def run(self):
        """
        Runs the "cautious algorithm" from the paper.
        """

        if self.__clusters__ is None:
            self.__clusters__ = []
            
            while len(self.__G_nodes__) > 0:
                # Make sure we try all the vertices until we run out
                vs = random.sample(self.__G_nodes__, len(self.__G_nodes__))

                Av = None

                for v in vs:
                    Av = self.positive_neighbours(v).copy()

                    # Vertex removal step
                    for x in self.positive_neighbours(v):
                        if not self.delta_good(x, Av, 3 * self.__delta__):
                            Av.remove(x)

                    # Vertex addition step
                    Y = set(y for y in self.__G_nodes__
                              if self.delta_good(y, Av, 7 * self.__delta__))
                    Av = Av | Y

                    if len(Av) > 0:
                        break

                # Second quit condition: all sets Av are empty
                if len(Av) == 0:
                    break

                self.__clusters__.append(Av)
                self.__remove_cluster__(Av)

            # add all remaining vertices as singleton clusters
            for v in self.__G_nodes__:
                self.__clusters__.append(set([v]))

        return self.__clusters__


def get_all_possible_pairs(sentences):
    """yields all possible pairs in sentence list"""
    pairs = []
    for i in range(len(sentences) - 1):
        for j in range(i + 1, len(sentences)):
            pairs.append((sentences[i], sentences[j]))

    return pairs


def prepare_pair(x, y, label, max_length=256):
    """returns input_ids, input_masks, labels for pair of sentences in BERT input format"""
    global tokenizer
    x = tokenizer.encode_plus(x, pad_to_max_length=True, add_special_tokens=True, max_length=max_length)
    y = tokenizer.encode_plus(y, pad_to_max_length=True, add_special_tokens=True, max_length=max_length)

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


def predict(sentences, threshold=0.7):
    """
    args:
    sentences = [ {"id": 1, "text":"text of sentence 1"}, {"id": 3, "text":"text of sentence 3"}, {"id": 4, "text":"text of sentence 4"}]
    
    returns:
    List of clusters
    """
    if len(sentences) < 2:
        return [sentences, ]

    pairs = []
    for p in get_all_possible_pairs(sentences):
        x, y = p
        _x, _y, _ = prepare_pair(x["text"], y["text"], -1)
        _x, _y = get_representation(_x), get_representation(_y)
        prediction = scorer_model(_x, _y).item()
        pairs.append((x, y, prediction))

    if len(pairs) == 1:
        if pairs[0][2] >= threshold:
            return [[pairs[0][0], pairs[0][1]],]

    graph = nx.Graph()
    for s in sentences:
        graph.add_node(s["id"])

    for x, y, p in pairs:
        if p >= threshold: 
            graph.add_edge(x["id"], y["id"], weights=p) 
    
    solver = Solver(graph)
    id_clusters = solver.run()
 
    sent_clusters = []
    for id_c in id_clusters:
        sent_clusters.append([ x for x in sentences if x["id"] in id_c ])

    return sent_clusters

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

scorer_model = Scorer()
scorer_model.load_state_dict(torch.load("scorer_model.pt"))
scorer_model = scorer_model.to(device)
scorer_model.eval()

if __name__ == "__main__":
    # copy scorer_model from /home/asafaya19/coreference-model/scorer_model.pt
    sents = [ 
        {"id": 1, "text" : "Another killing incident today in NYC"},
        {"id": 3, "text" : "one man killed in New York City "}, 
        {"id": 4, "text" : "all of the protesters condemned the irresponsible availability of guns"}
        ]
    print("Input sentences :\n", sents)
    print("Output clusters:\n", "\n".join(str(c) for c in predict(sents)))