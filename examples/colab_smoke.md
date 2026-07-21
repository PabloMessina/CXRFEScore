# Colab smoke test (human)

Run this **after** `cxrfescore` is uploaded to TestPyPI and **before** production PyPI.

## Setup

1. Open [Google Colab](https://colab.research.google.com/).
2. Runtime → Change runtime type → **GPU** (T4 is enough).
3. Run the cells below in order.

## Install from TestPyPI

```python
!pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "cxrfescore==0.2.3"
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

## Optional visualization (easy pair)

```python
!pip install "cxrfescore[viz]"
metric.visualize_fact_similarity(ref_report=refs[0], cand_report=hyps[0])
```

## Challenging / adversarial pairs

These cases are designed to stress clinical nuances that embedding metrics often mishandle
(or handle well). For each pair, inspect the printed score **and** the fact-similarity heatmap.
A good metric should score clinically wrong pairs lower than near-paraphrases.

```python
adversarial_cases = [
    {
        "name": "Negation flip (should score LOW)",
        "note": "Nearly identical wording, opposite clinical meaning.",
        "ref": "There is no evidence of pneumothorax.",
        "hyp": "There is evidence of pneumothorax.",
    },
    {
        "name": "Laterality swap (should score LOW)",
        "note": "Same finding, wrong side — clinically important error.",
        "ref": "Moderate right pleural effusion.",
        "hyp": "Moderate left pleural effusion.",
    },
    {
        "name": "Severity / size swap (should score MID-LOW)",
        "note": "Correct finding/side, wrong magnitude.",
        "ref": "Small right pleural effusion.",
        "hyp": "Large right pleural effusion.",
    },
    {
        "name": "Synonym paraphrase (should score HIGH)",
        "note": "Different wording, same meaning — robustness check.",
        "ref": "The heart is enlarged. There is pulmonary edema.",
        "hyp": "Cardiomegaly is present. Findings consistent with pulmonary oedema.",
    },
    {
        "name": "Presence vs absence of shared anatomy (should score LOW)",
        "note": "Shares anatomy tokens but asserts opposite finding.",
        "ref": "The lungs are clear without focal consolidation.",
        "hyp": "There is focal consolidation in the lungs.",
    },
    {
        "name": "Device / support device mismatch (should score LOW)",
        "note": "Support-device facts often confuse overlap-based metrics.",
        "ref": "Endotracheal tube tip is appropriately positioned above the carina.",
        "hyp": "Nasogastric tube tip is appropriately positioned in the stomach.",
    },
]
```

```python
# Score all adversarial pairs in one batch
adv_hyps = [c["hyp"] for c in adversarial_cases]
adv_refs = [c["ref"] for c in adversarial_cases]
adv_result = metric(adv_hyps, adv_refs)

print("=== Adversarial pair scores ===")
for case, score in zip(adversarial_cases, adv_result["per_pair_similarity"]):
    print(f"\n[{case['name']}] score={float(score):.4f}")
    print(f"  note: {case['note']}")
    print(f"  REF: {case['ref']}")
    print(f"  HYP: {case['hyp']}")
```

```python
# Visualize each challenging pair (one heatmap per case)
for case in adversarial_cases:
    print("\n" + "=" * 72)
    print(case["name"])
    print(case["note"])
    print("=" * 72)
    metric.visualize_fact_similarity(
        ref_report=case["ref"],
        cand_report=case["hyp"],
    )
```

```python
# Quick relative sanity check (not a hard unit test — inspect failures by eye)
scores = {c["name"]: float(s) for c, s in zip(adversarial_cases, adv_result["per_pair_similarity"])}
synonym = scores["Synonym paraphrase (should score HIGH)"]
negation = scores["Negation flip (should score LOW)"]
laterality = scores["Laterality swap (should score LOW)"]

print("synonym:", synonym)
print("negation:", negation)
print("laterality:", laterality)

if synonym <= negation:
    print("WARNING: synonym paraphrase did not outscore negation flip — inspect heatmaps.")
if synonym <= laterality:
    print("WARNING: synonym paraphrase did not outscore laterality swap — inspect heatmaps.")
if synonym > negation and synonym > laterality:
    print("Relative ordering looks reasonable (synonym > negation/laterality).")
```

## Checklist

- [ ] Install from TestPyPI succeeds
- [ ] Models download from Hugging Face
- [ ] CUDA used when GPU runtime is selected
- [ ] Similar pair scores higher than unrelated pair
- [ ] Adversarial pairs inspected (negation, laterality, severity, synonyms, devices)
- [ ] No import / runtime errors

## Citation

If you use CXRFEScore, please cite [Messina et al., Findings of ACL 2024](https://aclanthology.org/2024.findings-acl.236/). Full BibTeX is in the [README](../README.md#citation).
