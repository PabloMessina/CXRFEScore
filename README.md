# CXRFEScore

Fact-level embedding metric for evaluating chest X-ray radiology report generation.

CXRFEScore extracts factual statements from reports with a T5 fact extractor, embeds them with CXRFE (or compatible CXR-BERT models), and scores hypothesis/reference pairs via soft bipartite matching of fact embeddings.

Based on:

> Messina et al., *Extracting and Encoding: Leveraging Large Language Models and Medical Knowledge to Enhance Radiological Text Representation*, Findings of ACL 2024.

## Install

```bash
pip install cxrfescore
```

Optional visualization support:

```bash
pip install cxrfescore[viz]
```

**Python:** 3.10+

### Runtime dependencies (installed automatically)

| Package | Role |
|---|---|
| `torch` | Model inference |
| `transformers` | Load Hugging Face models |
| `numpy` | Arrays / aggregation |
| `scikit-learn` | Cosine similarity |
| `nltk` | Sentence splitting |
| `tqdm` | Progress bars |
| `platformdirs` | Default cache directory |

Optional: `matplotlib` via `cxrfescore[viz]` for `visualize_fact_similarity`.

### Additional requirements (not pip packages)

On first use the metric downloads:

- **Encoder (default):** [`pamessina/CXRFE`](https://huggingface.co/pamessina/CXRFE)
- **Fact extractor:** [`pamessina/T5FactExtractor`](https://huggingface.co/pamessina/T5FactExtractor)
- **NLTK** `punkt` / `punkt_tab` (auto-downloaded)

Supported alternate encoders:

- `microsoft/BiomedVLP-CXR-BERT-specialized`
- `microsoft/BiomedVLP-BioViL-T`

A GPU is recommended but not required. CPU works and is slower.

Known-good stack from development: PyTorch 2.x + `transformers` 4.x.

## Quick start

```python
from cxrfescore import CXRFEScore

metric = CXRFEScore(device="cuda")  # or "cpu"

hyps = [
    "There is a small right pleural effusion. The heart size is normal.",
]
refs = [
    "Small right pleural effusion. Normal heart size.",
]

result = metric(hyps, refs)
print(result["mean_similarity"])
print(result["per_pair_similarity"])
```

### Caching

By default, sentence→facts and fact→embedding maps are cached under the platform user cache directory for `cxrfescore` (via `platformdirs`).

```python
metric = CXRFEScore(use_cache=True, cache_dir="/path/to/cache")
# ... after scoring ...
metric.save_cache()
```

Disable with `use_cache=False`.

### Visualization

```python
# requires: pip install cxrfescore[viz]
metric.visualize_fact_similarity(ref_report=refs[0], cand_report=hyps[0])
```

## Try it on Google Colab

After publishing, install in Colab (GPU runtime recommended):

```python
!pip install cxrfescore
```

For a pre-release check against TestPyPI:

```python
!pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ cxrfescore
```

See [`examples/colab_smoke.md`](examples/colab_smoke.md) for a full smoke-test checklist.

## Citation

```bibtex
@inproceedings{messina-etal-2024-extracting,
    title = "Extracting and Encoding: Leveraging Large Language Models and Medical Knowledge to Enhance Radiological Text Representation",
    author = "Messina, Pablo and others",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
    year = "2024",
    url = "https://aclanthology.org/2024.findings-acl.236/"
}
```

Paper: [ACL Anthology](https://aclanthology.org/2024.findings-acl.236/) · [arXiv](https://arxiv.org/abs/2407.01948)

## Development

```bash
pip install -e ".[dev]"
pytest                          # unit + mocked tests (default)
pytest -m integration           # downloads HF models; needs network / GPU optional
python -m build
```

## License

MIT
