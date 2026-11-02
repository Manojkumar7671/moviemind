import json
import math
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.baselines import ALL_GENRES

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

def run_ab_simulation(
    data_dir: Path = DATA_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR,
    n_users: int = 5000,
    seed: int = 42,
) -> Dict:
    np.random.seed(seed)
    test_df = pd.read_parquet(data_dir / "test.parquet")
    train_df = pd.read_parquet(data_dir / "train.parquet")
    movies_df = pd.read_parquet(data_dir / "movies.parquet")
    mmr_df = pd.read_parquet(artifacts_dir / "mmr_candidates.parquet")

    # Load popular items for control
    pop_items = train_df["movie_id"].value_counts().index.tolist()
    user_train_hist = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    # Treatment recommendations from Full Pipeline (MMR candidates)
    treatment_recs = {int(r["user_id"]): list(r["candidate_ids"])[:10] for _, r in mmr_df.iterrows()}

    # Genre lookup for attraction modeling
    genre_map = {g: i for i, g in enumerate(ALL_GENRES)}
    movie_genres = {}
    for _, r in movies_df.iterrows():
        mid = int(r["movie_id"])
        vec = np.zeros(len(ALL_GENRES), dtype=np.float32)
        for g in str(r["genres"]).split("|"):
            if g in genre_map:
                vec[genre_map[g]] = 1.0
        norm = np.linalg.norm(vec)
        movie_genres[mid] = (vec / norm) if norm > 0 else vec

    # User historical genre preference from train
    user_pref = {}
    for uid, g in train_df[train_df["rating"] >= 3.5].groupby("user_id"):
        mids = g["movie_id"].values
        u_vec = np.zeros(len(ALL_GENRES), dtype=np.float32)
        for m in mids:
            if m in movie_genres:
                u_vec += movie_genres[m]
        norm = np.linalg.norm(u_vec)
        user_pref[uid] = (u_vec / norm) if norm > 0 else u_vec

    test_targets = dict(zip(test_df["user_id"], test_df["movie_id"]))
    sample_users = [u for u in test_df["user_id"].unique() if u in user_pref][:n_users]

    # Synthetic click simulation
    # Position bias: P(examine at rank r) = 1 / log2(r + 1)
    def simulate_clicks(recs: List[int], target_mid: int, u_vec: np.ndarray) -> int:
        clicks = 0
        for rank, mid in enumerate(recs[:10], start=1):
            p_exam = 1.0 / math.log2(rank + 1)
            # Attraction probability
            if mid == target_mid:
                p_attract = 0.90
            else:
                m_vec = movie_genres.get(mid, np.zeros(len(ALL_GENRES)))
                genre_aff = float(np.dot(u_vec, m_vec))
                p_attract = 0.05 + 0.35 * max(genre_aff, 0.0)

            p_click = p_exam * p_attract
            if np.random.rand() < p_click:
                clicks += 1
        return clicks

    # Split users 50/50: Control (A: Popularity) vs Treatment (B: Full Pipeline)
    n_sample = len(sample_users)
    half = n_sample // 2
    users_a = sample_users[:half]
    users_b = sample_users[half:]

    clicks_a = 0
    impressions_a = len(users_a) * 10
    sessions_with_click_a = 0

    for u in users_a:
        seen = user_train_hist.get(u, set())
        recs_a = [m for m in pop_items if m not in seen][:10]
        c = simulate_clicks(recs_a, test_targets[u], user_pref[u])
        clicks_a += c
        if c > 0:
            sessions_with_click_a += 1

    clicks_b = 0
    impressions_b = len(users_b) * 10
    sessions_with_click_b = 0

    for u in users_b:
        recs_b = treatment_recs.get(u, [])[:10]
        if not recs_b:
            seen = user_train_hist.get(u, set())
            recs_b = [m for m in pop_items if m not in seen][:10]
        c = simulate_clicks(recs_b, test_targets[u], user_pref[u])
        clicks_b += c
        if c > 0:
            sessions_with_click_b += 1

    ctr_a = clicks_a / impressions_a
    ctr_b = clicks_b / impressions_b
    relative_lift = ((ctr_b - ctr_a) / ctr_a) * 100.0

    session_ctr_a = sessions_with_click_a / len(users_a)
    session_ctr_b = sessions_with_click_b / len(users_b)
    session_lift = ((session_ctr_b - session_ctr_a) / session_ctr_a) * 100.0

    # 95% Confidence Interval for difference in item CTR
    p_pool = (clicks_a + clicks_b) / (impressions_a + impressions_b)
    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / impressions_a + 1.0 / impressions_b))
    diff = ctr_b - ctr_a
    ci_lower = diff - 1.96 * se
    ci_upper = diff + 1.96 * se
    z_stat = diff / se if se > 0 else 0
    # Two-tailed p-value via normal approximation
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_stat) / math.sqrt(2.0))))

    results = {
        "sample_size": n_sample,
        "control": {
            "name": "Popularity Baseline",
            "users": len(users_a),
            "impressions": impressions_a,
            "clicks": clicks_a,
            "ctr": round(ctr_a, 4),
            "session_ctr": round(session_ctr_a, 4),
        },
        "treatment": {
            "name": "Full Pipeline (Two-Tower + Ranker + MMR)",
            "users": len(users_b),
            "impressions": impressions_b,
            "clicks": clicks_b,
            "ctr": round(ctr_b, 4),
            "session_ctr": round(session_ctr_b, 4),
        },
        "relative_ctr_lift_pct": round(relative_lift, 2),
        "session_ctr_lift_pct": round(session_lift, 2),
        "ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
        "p_value": round(p_value, 6),
        "is_significant": bool(p_value < 0.05),
    }

    with open(artifacts_dir / "ab_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("=== Phase 10: Simulated A/B Test Results ===")
    print(f"Control (Popularity) CTR    : {ctr_a:.2%} (Sessions: {session_ctr_a:.2%})")
    print(f"Treatment (Full Pipeline) CTR: {ctr_b:.2%} (Sessions: {session_ctr_b:.2%})")
    print(f"Relative CTR Lift           : +{relative_lift:.2f}%")
    print(f"95% Confidence Interval     : [{ci_lower:+.4f}, {ci_upper:+.4f}]")
    print(f"p-value                     : {p_value:.2e} (Significant: {results['is_significant']})")
    return results

if __name__ == "__main__":
    run_ab_simulation()
