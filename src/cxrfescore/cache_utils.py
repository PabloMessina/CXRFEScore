"""Helpers for inspecting CXRFEScore on-disk caches."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

from cxrfescore.metric import DEFAULT_CACHE_DIR


def _format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.2f} KB"
    if num_bytes < 1024**3:
        return f"{num_bytes / 1024**2:.2f} MB"
    return f"{num_bytes / 1024**3:.2f} GB"


def _sample_entry(key: Any, value: Any) -> Dict[str, Any]:
    sample: Dict[str, Any] = {"key": key}
    if hasattr(value, "shape"):
        sample["value_kind"] = "array"
        sample["shape"] = tuple(value.shape)
        sample["dtype"] = str(getattr(value, "dtype", ""))
    else:
        sample["value_kind"] = type(value).__name__
        value_repr = repr(value)
        if len(value_repr) > 120:
            value_repr = value_repr[:120] + "..."
        sample["value_repr"] = value_repr
    return sample


def inspect_cache(
    cache_dir: Optional[Union[str, Path]] = None,
    max_samples: int = 5,
    *,
    print_report: bool = True,
) -> Dict[str, Any]:
    """
    Inspect CXRFEScore cache pickle files under ``cache_dir``.

    Looks for ``*.pkl`` recursively (sentence→facts at the root and
    fact→embedding files under ``embeddings/<encoder>/``).

    Args:
        cache_dir: Base cache directory. Defaults to the platform user cache
            dir for ``cxrfescore`` (same default as ``CXRFEScore``).
        max_samples: How many key/value samples to show per file.
        print_report: If True, print a human-readable report.

    Returns:
        A summary dict with ``cache_dir``, ``exists``, and a ``files`` list.
    """
    root = Path(cache_dir) if cache_dir is not None else Path(DEFAULT_CACHE_DIR)
    summary: Dict[str, Any] = {
        "cache_dir": str(root),
        "exists": root.exists(),
        "files": [],
    }

    if print_report:
        print("=" * 60)
        print(f"CXRFEScore cache: {root}")
        print("=" * 60)

    if not root.exists():
        if print_report:
            print("Cache directory does not exist yet.")
        return summary

    pkl_paths = sorted(root.rglob("*.pkl"))
    if not pkl_paths:
        if print_report:
            print("No .pkl cache files found.")
        return summary

    for path in pkl_paths:
        rel = str(path.relative_to(root))
        file_info: Dict[str, Any] = {
            "path": str(path),
            "relative_path": rel,
            "size_bytes": path.stat().st_size,
            "size_human": _format_size(path.stat().st_size),
            "entries": None,
            "error": None,
            "samples": [],
        }

        if print_report:
            print(f"\n--- {rel} ({file_info['size_human']}) ---")

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if not isinstance(data, dict):
                file_info["error"] = f"expected dict, got {type(data).__name__}"
                if print_report:
                    print(f"Unexpected type: {type(data).__name__}")
            else:
                file_info["entries"] = len(data)
                keys = list(data.keys())[:max_samples]
                for key in keys:
                    file_info["samples"].append(_sample_entry(key, data[key]))
                if print_report:
                    print(f"entries: {len(data)}")
                    for sample in file_info["samples"]:
                        if sample["value_kind"] == "array":
                            print(
                                f"  {sample['key']!r} -> array "
                                f"shape={sample['shape']} dtype={sample['dtype']}"
                            )
                        else:
                            print(f"  {sample['key']!r} -> {sample['value_repr']}")
                    if len(data) > max_samples:
                        print(f"  ... ({len(data) - max_samples} more)")
        except Exception as e:
            file_info["error"] = str(e)
            if print_report:
                print(f"failed to load: {e}")

        summary["files"].append(file_info)

    if print_report:
        print()
    return summary
