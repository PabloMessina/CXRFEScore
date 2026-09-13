# Minimal example — SRR-BERT-Leaves-with-Statuses

Uses CLS embeddings from [`StanfordAIMI/SRR-BERT-Leaves-with-Statuses`](https://huggingface.co/StanfordAIMI/SRR-BERT-Leaves-with-Statuses) (768-d). Present/uncertain/absent status labels are not used — only the BERT encoder CLS vector.

```python
!pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "cxrfescore==0.2.4"

from cxrfescore import CXRFEScore

metric = CXRFEScore(
    encoder_model_name="StanfordAIMI/SRR-BERT-Leaves-with-Statuses",
    device="cuda",
)
result = metric(
    ["There is a small right pleural effusion."],
    ["Small right pleural effusion."],
)
print(result["mean_similarity"])
metric.save_cache()
```
