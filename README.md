# event-coreference

## Coreference-model

```python

sents = [ 
    {"id": 1, "text" : "Another killing incident today in NYC"},
    {"id": 3, "text" : "one man killed in New York City "}, 
    {"id": 4, "text" : "all of the protesters condemned the irresponsible availability of guns"}
]

print("Input sentences :\n", sents)
print("Output clusters:\n", "\n".join(str(c) for c in predict(sents)))

```