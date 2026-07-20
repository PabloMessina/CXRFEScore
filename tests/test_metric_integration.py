"""Integration smoke tests that download real Hugging Face models.

Run explicitly:

    pytest -m integration

These are skipped by default (see pyproject.toml addopts).
"""

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def metric(tmp_path_factory):
    from cxrfescore import CXRFEScore

    cache_dir = tmp_path_factory.mktemp("cxrfescore_integration_cache")
    return CXRFEScore(
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        use_cache=True,
        cache_dir=str(cache_dir),
        batch_size=8,
        num_workers=0,
        verbose=True,
    )


def test_identical_reports_score_high(metric):
    report = (
        "There is a small right pleural effusion. The cardiomediastinal "
        "silhouette is within normal limits."
    )
    result = metric([report], [report])
    assert -1.0 <= result["mean_similarity"] <= 1.0
    assert result["mean_similarity"] > 0.8
    assert result["per_pair_similarity"].shape == (1,)


def test_unrelated_reports_score_lower(metric):
    hyp = "Large left pneumothorax with mediastinal shift."
    ref = "No acute cardiopulmonary process. Lungs are clear."
    identical = metric([hyp], [hyp])["mean_similarity"]
    unrelated = metric([hyp], [ref])["mean_similarity"]
    assert -1.0 <= unrelated <= 1.0
    assert identical > unrelated
