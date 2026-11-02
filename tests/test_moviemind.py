import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.baselines import ALL_GENRES, evaluate_recs
from src.coldstart import ColdStartRecommender
from src.diversity import mmr_rerank

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

# 1. Test: Split has no leakage
def test_split_has_no_leakage():
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    val = pd.read_parquet(DATA_DIR / "val.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")

    assert len(val) == 6040, f"Expected 6,040 val interactions, got {len(val)}"
    assert len(test) == 6040, f"Expected 6,040 test interactions, got {len(test)}"
    assert len(train) + len(val) + len(test) == 1000209

    # Max train timestamp < val timestamp < test timestamp
    max_train_ts = train.groupby("user_id")["timestamp"].max().to_dict()
    val_ts = dict(zip(val["user_id"], val["timestamp"]))
    test_ts = dict(zip(test["user_id"], test["timestamp"]))

    for uid in test_ts:
        t_train = max_train_ts[uid]
        t_val = val_ts[uid]
        t_test = test_ts[uid]
        assert t_train <= t_val, f"Leakage: user {uid} train timestamp {t_train} > val {t_val}"
        assert t_val <= t_test, f"Leakage: user {uid} val timestamp {t_val} > test {t_test}"

# 2. Test: Metrics calculation is exact
def test_metrics_correctness():
    # Synthetic recommendations and targets
    recs = {
        1: [10, 20, 30, 40, 50],
        2: [99, 88, 77, 66, 55],
    }
    targets = {
        1: 20,  # Rank 2 (1-based)
        2: 999, # Not in top-5
    }
    train_counts = {10: 100, 20: 50, 30: 25, 40: 10, 50: 5, 99: 10, 88: 10, 77: 10, 66: 10, 55: 10}
    # Mock genre vectors
    g_vecs = {m: np.zeros(len(ALL_GENRES)) for m in [10, 20, 30, 40, 50, 99, 88, 77, 66, 55]}
    g_vecs[10][0] = 1.0  # Action
    g_vecs[20][1] = 1.0  # Adventure
    g_vecs[30][2] = 1.0  # Animation
    g_vecs[40][3] = 1.0
    g_vecs[50][4] = 1.0

    m = evaluate_recs(recs, targets, train_counts, total_catalog_size=100, genre_vectors=g_vecs, ks=(5,))

    # Recall@5: user 1 has 1.0, user 2 has 0.0 -> mean 0.5
    assert m["Recall@5"] == 0.5
    # NDCG@5: user 1 has 1 / log2(2 + 1) = 1 / 1.5849625 ~ 0.63093, user 2 has 0 -> mean ~ 0.31546
    expected_ndcg = 0.5 * (1.0 / math.log2(3))
    assert abs(m["NDCG@5"] - expected_ndcg) < 1e-4
    # Coverage@5: 10 unique items out of 100 catalog -> 0.10
    assert m["Coverage@5"] == 0.10

# 3. Test: MMR diversity >= no-MMR
def test_mmr_diversity_greater_or_equal():
    movies_df = pd.read_parquet(DATA_DIR / "movies.parquet")
    from src.baselines import get_genre_matrix
    genre_vectors = get_genre_matrix(movies_df)

    # Let item_ids be 10 movies, some with identical genres
    item_ids = list(movies_df["movie_id"].head(20).values)
    scores = [1.0 - i * 0.03 for i in range(len(item_ids))]

    # Pure relevance (no MMR penalty)
    pure_recs = mmr_rerank(item_ids, scores, genre_vectors, top_k=10, lmbda=1.0)
    # Diversified MMR
    div_recs = mmr_rerank(item_ids, scores, genre_vectors, top_k=10, lmbda=0.3)

    # Compute ILD of both
    def calc_ild(item_list):
        vecs = np.array([genre_vectors.get(m, np.zeros(len(ALL_GENRES))) for m in item_list])
        sims = np.dot(vecs, vecs.T)
        n = len(item_list)
        return float(np.mean(1.0 - sims[np.triu_indices(n, k=1)]))

    ild_pure = calc_ild(pure_recs)
    ild_div = calc_ild(div_recs)
    assert ild_div >= ild_pure - 1e-6, f"Expected ILD(div) >= ILD(pure), got {ild_div} vs {ild_pure}"

# 4. Test: Cold-start returns exactly n items with proper schema
def test_coldstart_returns_n_items():
    cs = ColdStartRecommender(DATA_DIR, ARTIFACTS_DIR)
    for n in (3, 5, 10):
        recs = cs.recommend(["Action", "Sci-Fi", "Thriller"], top_n=n)
        assert len(recs) == n, f"Expected {n} recommendations, got {len(recs)}"
        for r in recs:
            assert "movie_id" in r and isinstance(r["movie_id"], int)
            assert "title" in r and isinstance(r["title"], str)
            assert "genres" in r and isinstance(r["genres"], str)
            assert "year" in r and isinstance(r["year"], int)
            assert "score" in r and isinstance(r["score"], float)

# 5. Test: API smoke test
def test_api_smoke_test():
    client = TestClient(app)

    # Health
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    # Genres
    r = client.get("/api/genres")
    assert r.status_code == 200
    assert len(r.json()["genres"]) == 18

    # Onboarding
    r = client.post("/api/onboarding", json={"user_id": "test_pytest", "genres": ["Action", "Comedy"]})
    assert r.status_code == 200
    assert len(r.json()["recommendations"]) == 6

    # Chat
    r = client.post("/api/chat", json={"user_id": "test_pytest", "message": "Search for Star Wars"})
    assert r.status_code == 200
    assert "Star Wars" in r.json()["response"]

    # Feedback
    r = client.post("/api/feedback", json={"user_id": "test_pytest", "movie_id": 260, "feedback_type": "like"})
    assert r.status_code == 200
    assert r.json()["status"] == "success"

    # AB
    r = client.get("/api/ab")
    assert r.status_code == 200
    assert "relative_ctr_lift_pct" in r.json()

    # Frontend routes
    r = client.get("/")
    assert r.status_code == 200
    assert "MovieMind" in r.text

    r = client.get("/ab")
    assert r.status_code == 200
    assert "A/B Test" in r.text
