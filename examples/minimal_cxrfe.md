# Minimal example — CXRFE (default)

```python
!pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "cxrfescore==0.2.2"

from cxrfescore import CXRFEScore

metric = CXRFEScore(encoder_model_name="pamessina/CXRFE", device="cuda")
result = metric(
    ["There is a small right pleural effusion."],
    ["Small right pleural effusion."],
)
print(result["mean_similarity"])
metric.save_cache()  # persist in-memory caches to disk when done
```
