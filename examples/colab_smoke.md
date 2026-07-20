# Colab smoke test (human)

Run this **after** `cxrfescore` is uploaded to TestPyPI and **before** production PyPI.

## Setup

1. Open [Google Colab](https://colab.research.google.com/).
2. Runtime → Change runtime type → **GPU** (T4 is enough).
3. Run the cells below in order.

## Install from TestPyPI

```python
!pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "cxrfescore==0.1.0"
```

After the production release, use:

```python
!pip install cxrfescore
```

## Smoke test

```python
import torch
from cxrfescore import CXRFEScore

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

metric = CXRFEScore(
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=8,
    num_workers=0,
    verbose=True,
)

hyps = [
    "There is a small right pleural effusion. The heart size is normal.",
    "Large left pneumothorax with mediastinal shift.",
]
refs = [
    "Small right pleural effusion. Normal heart size.",
    "No acute cardiopulmonary process. Lungs are clear.",
]

result = metric(hyps, refs)
print("mean_similarity:", result["mean_similarity"])
print("per_pair_similarity:", result["per_pair_similarity"])

# Sanity checks
assert result["per_pair_similarity"].shape == (2,)
assert result["per_pair_similarity"][0] > result["per_pair_similarity"][1]
print("Colab smoke test PASSED")
```

## Optional visualization

```python
!pip install "cxrfescore[viz]"
metric.visualize_fact_similarity(ref_report=refs[0], cand_report=hyps[0])
```

## Checklist

- [ ] Install from TestPyPI succeeds
- [ ] Models download from Hugging Face
- [ ] CUDA used when GPU runtime is selected
- [ ] Similar pair scores higher than unrelated pair
- [ ] No import / runtime errors
