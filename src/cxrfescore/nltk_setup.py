"""Ensure required NLTK tokenizers are available."""

import nltk

REQUIRED_RESOURCES = ["punkt", "punkt_tab"]


def ensure_nltk_resources():
    """Download missing NLTK resources needed for sentence tokenization."""
    missing_resources = []
    for resource in REQUIRED_RESOURCES:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            missing_resources.append(resource)

    if not missing_resources:
        return

    print(f"Downloading missing NLTK resources: {missing_resources}...")
    try:
        for resource in missing_resources:
            nltk.download(resource, quiet=True)
        for resource in missing_resources:
            nltk.data.find(f"tokenizers/{resource}")
        print("NLTK resources downloaded successfully.")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        print(f"  python -m nltk.downloader {' '.join(missing_resources)}")
