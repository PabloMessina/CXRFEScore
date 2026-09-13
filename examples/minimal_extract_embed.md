# Minimal example — extract facts / embed facts only

`extract_facts` treats each string as a **full report** (sentence-split → extract → aggregate).
`embed_facts` embeds already-extracted fact strings.

```python
!pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "cxrfescore==0.2.4"

from cxrfescore import CXRFEScore

metric = CXRFEScore(device="cuda")

reports = [
    "There is a small right pleural effusion. The heart size is normal.",
]
facts_per_report = metric.extract_facts(reports)
print(facts_per_report)

# Embed the facts from the first report
embeddings = metric.embed_facts(facts_per_report[0])
print(embeddings.shape)  # (num_facts, embedding_dim)

metric.save_cache()
```
