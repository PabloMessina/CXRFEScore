"""CXRFEScore: fact-level embedding metric for chest X-ray report evaluation."""

from cxrfescore.cache_utils import inspect_cache
from cxrfescore.metric import CXRFEScore

__all__ = ["CXRFEScore", "inspect_cache"]
__version__ = "0.2.4"
