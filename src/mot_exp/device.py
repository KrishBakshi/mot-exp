"""Auto-select CUDA → MPS → CPU for local inference."""

from __future__ import annotations


def select_device() -> str:
    """Return the best available torch device string."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
