try:
    import torch

    available = True
except ModuleNotFoundError:
    available = False

if available:
    from ._pytorch import *


def ensure_available():
    if not available:
        raise ModuleNotFoundError(
            "PyTorch is not installed and conversion functionality is not available"
        )
