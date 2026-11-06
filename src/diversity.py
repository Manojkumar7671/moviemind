from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.baselines import ALL_GENRES, evaluate_recs, get_genre_matrix

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

def mmr_rerank(
    item_ids: List[int],
    scores: List[float],
    genre_vectors: Dict[int, np.ndarray],
    top_k: int = 50,
    lmbda: float = 0.7,
) -> List[int]:
    """Maximal Marginal Relevance (MMR) re-ranking balancing relevance and diversity."""
    n = len(item_ids)
    if n <= top_k or lmbda >= 1.0:
        return list(item_ids[:top_k])

    score_arr = np.array(scores, dtype=np.float32)
    s_min, s_max = float(score_arr.min()), float(score_arr.max())
    norm_scores = (score_arr - s_min) / (s_max - s_min + 1e-8)

    # Genre representation matrix
    vec_list = [genre_vectors.get(mid, np.zeros(len(ALL_GENRES), dtype=np.float32)) for mid in item_ids]
    mat = np.array(vec_list, dtype=np.float32)
    sim_mat = np.dot(mat, mat.T)

    selected = [int(np.argmax(norm_scores))]
    candidates = set(range(n)) - {selected[0]}

    while len(selected) < top_k and candidates:
        cand_list = list(candidates)
        # Max similarity to any already-selected item
        sub_sims = sim_mat[cand_list][:, selected]
        max_sims = np.max(sub_sims, axis=1)

        mmr_vals = lmbda * norm_scores[cand_list] - (1.0 - lmbda) * max_sims
        best_idx = cand_list[int(np.argmax(mmr_vals))]
        selected.append(best_idx)
        candidates.remove(best_idx)

    return [item_ids[i] for i in selected]

def run_diversity_analysis(
    data_dir: Path = DATA_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR,
    lambdas: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    default_lambda: float = 0.7,
):
    ranked_df = pd.read_parquet(artifacts_dir / "ranked_candidates.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    movies_df = pd.read_parquet(data_dir / "movies.parquet")
    train_df = pd.read_parquet(data_dir / "train.parquet")

    genre_vectors = get_genre_matrix(movies_df)
    test_targets = dict(zip(test_df["user_id"], test_df["movie_id"]))
    train_counts = train_df["movie_id"].value_counts().to_dict()

    print("=== Phase 5: MMR Accuracy vs. Diversity Tradeoff Analysis ===")
    records = []

    for lmb in lambdas:
        recs_dict = {}
        for _, row in ranked_df.iterrows():
            uid = int(row["user_id"])
            c_ids = row["candidate_ids"]
            c_scores = row["candidate_scores"]
            if len(c_ids) == 0:
                recs_dict[uid] = []
                continue
            # Re-rank from top 100 candidates to get 50
            recs_dict[uid] = mmr_rerank(c_ids[:100], c_scores[:100], genre_vectors, top_k=50, lmbda=lmb)

        metrics = evaluate_recs(
            recommendations=recs_dict,
            targets=test_targets,
            train_counts=train_counts,
            total_catalog_size=len(movies_df),
            genre_vectors=genre_vectors,
            ks=(10, 50),
        )
        rec = {"lambda": lmb, **metrics}
        records.append(rec)
        print(f"lambda={lmb:3.1f} | Recall@10={rec['Recall@10']:.4f} | NDCG@10={rec['NDCG@10']:.4f} | ILD@10={rec['ILD@10']:.4f} | Novelty@10={rec['Novelty@10']:.4f}")

    tradeoff_df = pd.DataFrame(records)
    tradeoff_df.to_parquet(artifacts_dir / "diversity_tradeoff.parquet", index=False)

    # Save default lambda candidates for serving pipeline
    default_recs = []
    for _, row in ranked_df.iterrows():
        uid = int(row["user_id"])
        c_ids = row["candidate_ids"]
        c_scores = row["candidate_scores"]
        reranked = mmr_rerank(c_ids[:100], c_scores[:100], genre_vectors, top_k=50, lmbda=default_lambda)
        default_recs.append({"user_id": uid, "candidate_ids": reranked})
    pd.DataFrame(default_recs).to_parquet(artifacts_dir / "mmr_candidates.parquet", index=False)
    print(f"\nDefault MMR (lambda={default_lambda}) saved to artifacts/mmr_candidates.parquet")
    return tradeoff_df

if __name__ == "__main__":
    run_diversity_analysis()
