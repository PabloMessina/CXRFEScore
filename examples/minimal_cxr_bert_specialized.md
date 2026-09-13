# Minimal example — CXR-BERT specialized

```python
!pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "cxrfescore==0.2.4"

from cxrfescore import CXRFEScore

metric = CXRFEScore(
    encoder_model_name="microsoft/BiomedVLP-CXR-BERT-specialized",
    device="cuda",
)
result = metric(
    ["There is a small right pleural effusion."],
    ["Small right pleural effusion."],
)
print(result["mean_similarity"])
metric.save_cache()
```
