import os
import torch


def get_device(prefer: str = None) -> str:
    """Pick the inference device: explicit argument or PARTCRAFTER_DEVICE, else cuda, mps, cpu."""
    prefer = prefer or os.environ.get("PARTCRAFTER_DEVICE")
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
