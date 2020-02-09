import json

def get_instance_pairs(file_name):
    """yields all possible pairs in a document"""
    with open(file_name, "r") as fi:
        instances = json.loads(fi.read())

    for events in instances:
        if len(events) < 1 or (len(events) == 1 and len(events[0]) < 2):
            continue
        
        positive_pairs = []
        for sentences in events:
            for i in range(len(sentences) - 1):
                for j in range(i + 1, len(sentences)):
                    positive_pairs.append((sentences[i], sentences[j]))
        
        negative_pairs = []
        for i in range(len(events) - 1):
            for j in range(i + 1, len(events)):
                for s_1 in events[i]:
                    negative_pairs += [ (s_1, s_2) for s_2 in events[j] ]

        if len(positive_pairs + negative_pairs) == 0:
            continue 

        yield positive_pairs, negative_pairs


def get_all_possible_pairs(sentences):
    """yields all possible pairs in sentence list"""
    pairs = []
    for i in range(len(sentences) - 1):
        for j in range(i + 1, len(sentences)):
            pairs.append((sentences[i], sentences[j]))

    return pairs