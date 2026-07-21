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
- `StanfordAIMI/SRR-BERT-Leaves` (CLS embeddings only; label head unused)

A GPU is recommended but not required. CPU works and is slower.

Known-good stack from development: PyTorch 2.x + `transformers` 4.x / 5.x.

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

### Extract facts or embeddings only

`extract_facts` treats each input string as a **full report**: sentence-split → T5 fact extraction → unique-fact aggregation (same pipeline as scoring).

```python
facts_per_report = metric.extract_facts(hyps)
embeddings = metric.embed_facts(facts_per_report[0])  # shape (num_facts, dim)
```

### Caching

With `use_cache=True` (default), results are kept **in memory** during the session:

- Sentence→facts: `{cache_dir}/sent_to_facts.pkl` (shared across encoders)
- Fact→embedding: `{cache_dir}/embeddings/<encoder_name>/fact_to_embedding.pkl` (per encoder)

Disk writes are **not** automatic. Call `save_cache()` explicitly after a large batch or at the end of a job:

```python
metric = CXRFEScore(use_cache=True, cache_dir="/path/to/cache")
# ... score / extract / embed ...
metric.save_cache()
```

Disable with `use_cache=False`.

Inspect what is already on disk:

```python
from cxrfescore import inspect_cache

inspect_cache()                 # default cache location
# or: metric.inspect_cache()    # uses this instance's cache_dir
```

### Visualization

```python
# requires: pip install cxrfescore[viz]
metric.visualize_fact_similarity(ref_report=refs[0], cand_report=hyps[0])
```

## Examples

| File | Purpose |
|---|---|
| [`examples/colab_smoke.md`](examples/colab_smoke.md) | Full Colab smoke + adversarial pairs |
| [`examples/minimal_cxrfe.md`](examples/minimal_cxrfe.md) | Default CXRFE encoder |
| [`examples/minimal_cxr_bert_specialized.md`](examples/minimal_cxr_bert_specialized.md) | CXR-BERT specialized |
| [`examples/minimal_biovil_t.md`](examples/minimal_biovil_t.md) | BioViL-T |
| [`examples/minimal_srr_bert_leaves.md`](examples/minimal_srr_bert_leaves.md) | SRR-BERT-Leaves CLS |
| [`examples/minimal_extract_embed.md`](examples/minimal_extract_embed.md) | `extract_facts` / `embed_facts` |

## Try it on Google Colab

After publishing, install in Colab (GPU runtime recommended):

```python
!pip install cxrfescore
```

For a pre-release check against TestPyPI:

```python
!pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "cxrfescore==0.2.2"
```

## Citation

If you use CXRFEScore, please cite:

> Pablo Messina, Rene Vidal, Denis Parra, Alvaro Soto, and Vladimir Araujo. 2024. [Extracting and Encoding: Leveraging Large Language Models and Medical Knowledge to Enhance Radiological Text Representation](https://aclanthology.org/2024.findings-acl.236/). In *Findings of the Association for Computational Linguistics: ACL 2024*, pages 3955–3986, Bangkok, Thailand. Association for Computational Linguistics.

```bibtex
@inproceedings{messina-etal-2024-extracting,
    title = "Extracting and Encoding: Leveraging Large Language Models and Medical Knowledge to Enhance Radiological Text Representation",
    author = "Messina, Pablo  and
      Vidal, Rene  and
      Parra, Denis  and
      Soto, Alvaro  and
      Araujo, Vladimir",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.236/",
    doi = "10.18653/v1/2024.findings-acl.236",
    pages = "3955--3986"
}
```

Paper: [ACL Anthology](https://aclanthology.org/2024.findings-acl.236/) · [PDF](https://aclanthology.org/2024.findings-acl.236.pdf) · [arXiv](https://arxiv.org/abs/2407.01948)

## Development

```bash
pip install -e ".[dev]"
pytest                          # unit + mocked tests (default)
pytest -m integration           # downloads HF models; needs network / GPU optional
python -m build
```

## License

MIT
