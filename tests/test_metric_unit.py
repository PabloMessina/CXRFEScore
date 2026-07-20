"""Unit tests for CXRFEScore with mocked models (no HF downloads)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cxrfescore.metric import CXRFEScore, DEFAULT_CACHE_DIR


def _make_metric(tmp_path, use_cache=True, embedding_dim=128):
    """Construct CXRFEScore without loading real HF models."""
    with (
        patch("cxrfescore.metric.ensure_nltk_resources"),
        patch("cxrfescore.metric.AutoTokenizer") as mock_enc_tok,
        patch("cxrfescore.metric.AutoModel") as mock_enc_model,
        patch("cxrfescore.metric.T5TokenizerFast") as mock_ext_tok,
        patch("cxrfescore.metric.T5ForConditionalGeneration") as mock_ext_model,
    ):
        encoder = MagicMock()
        encoder.eval = MagicMock()
        encoder.to = MagicMock(return_value=encoder)
        mock_enc_model.from_pretrained.return_value = encoder
        mock_enc_tok.from_pretrained.return_value = MagicMock()

        extractor = MagicMock()
        extractor.eval = MagicMock()
        extractor.to = MagicMock(return_value=extractor)
        mock_ext_model.from_pretrained.return_value = extractor
        mock_ext_tok.from_pretrained.return_value = MagicMock()

        metric = CXRFEScore(
            device="cpu",
            use_cache=use_cache,
            cache_dir=str(tmp_path / "cache"),
            batch_size=8,
            num_workers=0,
            verbose=False,
        )
        metric.encoder_model = encoder
        metric.extractor_model = extractor
        metric.embedding_dimension = embedding_dim
        return metric


def test_unsupported_encoder_raises(tmp_path):
    with (
        patch("cxrfescore.metric.ensure_nltk_resources"),
        patch("cxrfescore.metric.AutoTokenizer"),
        patch("cxrfescore.metric.AutoModel"),
        patch("cxrfescore.metric.T5TokenizerFast"),
        patch("cxrfescore.metric.T5ForConditionalGeneration"),
    ):
        with pytest.raises(ValueError, match="Unsupported model name"):
            CXRFEScore(
                encoder_model_name="not/a-real-model",
                device="cpu",
                use_cache=False,
            )


def test_default_cache_dir_is_platformdirs():
    assert "cxrfescore" in DEFAULT_CACHE_DIR


def test_length_mismatch_raises(tmp_path):
    metric = _make_metric(tmp_path, use_cache=False)
    with pytest.raises(ValueError, match="must be the same"):
        metric.compute(["a"], ["b", "c"])


def test_aggregate_facts_fallback_to_sentence():
    sents = ["No acute findings."]
    sent_to_facts = {"No acute findings.": []}
    facts = CXRFEScore._aggregate_facts(sents, sent_to_facts)
    assert facts == ["No acute findings."]


def test_aggregate_facts_dedup():
    sents = ["A.", "B."]
    sent_to_facts = {"A.": ["x", "y"], "B.": ["y", "z"]}
    facts = CXRFEScore._aggregate_facts(sents, sent_to_facts)
    assert facts == ["x", "y", "z"]


def test_empty_reports_score_zero(tmp_path):
    metric = _make_metric(tmp_path, use_cache=False)
    with patch("cxrfescore.metric.sent_tokenize", side_effect=lambda r: []):
        result = metric.compute([""], [""])
    assert result["mean_similarity"] == 0.0
    assert result["per_pair_similarity"].tolist() == [0.0]


def test_compute_scoring_math(tmp_path):
    """Identical unit embeddings → score near 1; orthogonal → near 0."""
    metric = _make_metric(tmp_path, use_cache=False, embedding_dim=2)

    def fake_extract(sentences, batch_size=None, num_workers=None):
        return [[s] for s in sentences]

    emb_table = {
        "same fact": np.array([1.0, 0.0], dtype=np.float32),
        "other fact": np.array([0.0, 1.0], dtype=np.float32),
    }

    def fake_embed(texts, batch_size=None):
        return np.stack([emb_table[t] for t in texts])

    metric._extract_facts_batch = fake_extract
    metric._get_embeddings_batch = fake_embed

    with patch(
        "cxrfescore.metric.sent_tokenize",
        side_effect=lambda r: [r] if r else [],
    ):
        identical = metric.compute(["same fact"], ["same fact"])
        orthogonal = metric.compute(["same fact"], ["other fact"])

    assert identical["mean_similarity"] == pytest.approx(1.0, abs=1e-5)
    assert orthogonal["mean_similarity"] == pytest.approx(0.0, abs=1e-5)


def test_cache_save_and_load(tmp_path):
    metric = _make_metric(tmp_path, use_cache=True)
    metric.sent_to_facts_cache["hello"] = ["fact a"]
    metric.fact_to_embedding_cache["fact a"] = np.ones(128, dtype=np.float32)
    metric.save_cache()

    metric2 = _make_metric(tmp_path, use_cache=True)
    assert metric2.sent_to_facts_cache["hello"] == ["fact a"]
    np.testing.assert_array_equal(
        metric2.fact_to_embedding_cache["fact a"], np.ones(128, dtype=np.float32)
    )


def test_visualize_requires_matplotlib(tmp_path):
    metric = _make_metric(tmp_path, use_cache=False)
    with patch.dict("sys.modules", {"matplotlib": None, "matplotlib.pyplot": None}):
        # Force ImportError path by making import fail
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("matplotlib"):
                raise ImportError("mocked missing matplotlib")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(ImportError, match="cxrfescore\\[viz\\]"):
                metric.visualize_fact_similarity("ref", "cand")
