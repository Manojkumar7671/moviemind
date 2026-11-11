#!/usr/bin/env python3
"""scripts/export_ranker_numpy.py

Exports PyTorch Wide&Deep checkpoint to NumPy .npz archive for zero-torch serve-time ranking.
"""

from pathlib import Path
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"


def export_wide_and_deep(
    pt_path: Path = ARTIFACTS_DIR / "wide_and_deep.pt",
    npz_path: Path = ARTIFACTS_DIR / "wide_and_deep_weights.npz",
):
    if not pt_path.exists():
        raise FileNotFoundError(f"Source PyTorch checkpoint not found: {pt_path}")

    sd = torch.load(pt_path, map_location="cpu")
    weights = {k: v.cpu().numpy() for k, v in sd.items()}
    np.savez_compressed(npz_path, **weights)
    print(f"Exported {len(weights)} weight tensors from {pt_path.name} -> {npz_path.name}")


if __name__ == "__main__":
    export_wide_and_deep()
