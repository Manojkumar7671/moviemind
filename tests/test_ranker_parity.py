"""tests/test_ranker_parity.py

Verifies mathematical parity between PyTorch WideAndDeepRanker and
pure NumPy NumpyWideAndDeepRanker (max abs diff < 1e-5 on 1,000 random inputs).
"""

from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed in runtime environment")

from src.ranker import NumpyWideAndDeepRanker, WideAndDeepRanker

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


@pytest.mark.torch
def test_numpy_wide_and_deep_parity():
    pt_path = ARTIFACTS_DIR / "wide_and_deep.pt"
    npz_path = ARTIFACTS_DIR / "wide_and_deep_weights.npz"

    assert pt_path.exists(), f"Missing {pt_path}"
    assert npz_path.exists(), f"Missing {npz_path}"

    state_dict = torch.load(pt_path, map_location="cpu")
    num_users = state_dict["user_emb.weight"].shape[0] - 1
    num_items = state_dict["item_emb.weight"].shape[0] - 1

    torch_ranker = WideAndDeepRanker(num_users=num_users, num_items=num_items)
    torch_ranker.load_state_dict(state_dict)
    torch_ranker.eval()

    numpy_ranker = NumpyWideAndDeepRanker.load(npz_path)

    np.random.seed(42)
    n_samples = 1000
    u_ids = np.random.randint(1, num_users + 1, size=n_samples, dtype=np.int64)
    i_ids = np.random.randint(1, num_items + 1, size=n_samples, dtype=np.int64)
    dense_feats = np.random.randn(n_samples, 4).astype(np.float32)

    with torch.no_grad():
        torch_out = torch_ranker(
            torch.tensor(u_ids, dtype=torch.long),
            torch.tensor(i_ids, dtype=torch.long),
            torch.tensor(dense_feats, dtype=torch.float32),
        ).cpu().numpy()

    numpy_out = numpy_ranker.score(u_ids, i_ids, dense_feats)

    max_abs_diff = float(np.max(np.abs(torch_out - numpy_out)))
    print(f"Max absolute difference on {n_samples} samples: {max_abs_diff:.4e}")

    assert max_abs_diff < 1e-5, f"Parity violated: max abs diff {max_abs_diff} >= 1e-5"
