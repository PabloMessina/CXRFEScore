"""Tests for inspect_cache helper."""

import pickle
from pathlib import Path

import numpy as np

from cxrfescore import inspect_cache
from cxrfescore.metric import _safe_model_dirname


def test_inspect_cache_missing_dir(tmp_path, capsys):
    missing = tmp_path / "nope"
    summary = inspect_cache(missing)
    assert summary["exists"] is False
    assert summary["files"] == []
    captured = capsys.readouterr().out
    assert "does not exist" in captured


def test_inspect_cache_reports_facts_and_embeddings(tmp_path, capsys):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    facts = {"sent a": ["fact a"]}
    embeds = {"fact a": np.ones(4, dtype=np.float32)}

    with open(cache_dir / "sent_to_facts.pkl", "wb") as f:
        pickle.dump(facts, f)

    embed_dir = cache_dir / "embeddings" / _safe_model_dirname("pamessina/CXRFE")
    embed_dir.mkdir(parents=True)
    with open(embed_dir / "fact_to_embedding.pkl", "wb") as f:
        pickle.dump(embeds, f)

    summary = inspect_cache(cache_dir, max_samples=3)
    assert summary["exists"] is True
    assert len(summary["files"]) == 2
    by_rel = {item["relative_path"]: item for item in summary["files"]}
    assert by_rel["sent_to_facts.pkl"]["entries"] == 1
    embed_rel = f"embeddings/{_safe_model_dirname('pamessina/CXRFE')}/fact_to_embedding.pkl"
    assert by_rel[embed_rel]["entries"] == 1
    assert by_rel[embed_rel]["samples"][0]["value_kind"] == "array"

    out = capsys.readouterr().out
    assert "sent_to_facts.pkl" in out
    assert "fact_to_embedding.pkl" in out


def test_inspect_cache_can_suppress_print(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    with open(cache_dir / "sent_to_facts.pkl", "wb") as f:
        pickle.dump({}, f)
    summary = inspect_cache(cache_dir, print_report=False)
    assert summary["files"][0]["entries"] == 0
