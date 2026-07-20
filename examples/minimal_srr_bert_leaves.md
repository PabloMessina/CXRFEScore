# Minimal example — SRR-BERT-Leaves

Uses CLS embeddings from [`StanfordAIMI/SRR-BERT-Leaves`](https://huggingface.co/StanfordAIMI/SRR-BERT-Leaves) (768-d). Label prediction heads are not used.

```python
!pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "cxrfescore==0.2.1"

from cxrfescore import CXRFEScore

metric = CXRFEScore(
    encoder_model_name="StanfordAIMI/SRR-BERT-Leaves",
    device="cuda",
)
result = metric(
    ["There is a small right pleural effusion."],
    ["Small right pleural effusion."],
)
print(result["mean_similarity"])
metric.save_cache()
```
