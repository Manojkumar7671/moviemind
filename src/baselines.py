import math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Standard list of all 18 MovieLens genres
ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

def get_genre_matrix(movies_df: pd.DataFrame) -> Dict[int, np.ndarray]:
    """Map movie_id to normalized multi-hot genre vector."""
    genre_to_idx = {g: i for i, g in enumerate(ALL_GENRES)}
    res = {}
    for _, row in movies_df.iterrows():
        vec = np.zeros(len(ALL_GENRES), dtype=np.float32)
        for g in str(row["genres"]).split("|"):
            if g in genre_to_idx:
                vec[genre_to_idx[g]] = 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        res[int(row["movie_id"])] = vec
    return res

def evaluate_recs(
    recommendations: Dict[int, List[int]],
    targets: Dict[int, int],
    train_counts: Dict[int, int],
    total_catalog_size: int,
    genre_vectors: Dict[int, np.ndarray],
    ks: Tuple[int, ...] = (10, 50),
) -> Dict[str, float]:
    """Compute Recall@K, NDCG@K, Coverage@K, Novelty@K, and Intra-List Diversity (ILD)@K."""
    results = {}
    total_train_events = max(sum(train_counts.values()), 1)
    # Compute self-information -log2(p(i)) per item
    self_info = {
        item: -math.log2(count / total_train_events)
        for item, count in train_counts.items()
        if count > 0
    }
    # Default for unseen items
    min_prob = 1.0 / (total_train_events + 1)
    default_self_info = -math.log2(min_prob)

    num_users = len(targets)
    if num_users == 0:
        return {}

    for k in ks:
        recalls = []
        ndcgs = []
        user_novelties = []
        user_ilds = []
        unique_recs = set()

        for u, target in targets.items():
            recs = recommendations.get(u, [])[:k]
            unique_recs.update(recs)

            # Recall & NDCG (leave-one-out)
            if target in recs:
                rank = recs.index(target) + 1  # 1-based rank
                recalls.append(1.0)
                ndcgs.append(1.0 / math.log2(rank + 1))
            else:
                recalls.append(0.0)
                ndcgs.append(0.0)

            # Novelty: mean self-information of recommended items
            if recs:
                nov = sum(self_info.get(m, default_self_info) for m in recs) / len(recs)
                user_novelties.append(nov)

            # Intra-List Diversity: mean pairwise distance (1 - cos_sim)
            if len(recs) > 1:
                vecs = [genre_vectors.get(m, np.zeros(len(ALL_GENRES))) for m in recs]
                mat = np.array(vecs)
                # Pairwise cosine similarity since vectors are L2-normalized
                sim_matrix = np.dot(mat, mat.T)
                n = len(recs)
                upper_tri = np.triu_indices(n, k=1)
                distances = 1.0 - sim_matrix[upper_tri]
                user_ilds.append(float(np.mean(distances)))
            else:
                user_ilds.append(0.0)

        results[f"Recall@{k}"] = float(np.mean(recalls))
        results[f"NDCG@{k}"] = float(np.mean(ndcgs))
        results[f"Coverage@{k}"] = len(unique_recs) / total_catalog_size
        results[f"Novelty@{k}"] = float(np.mean(user_novelties)) if user_novelties else 0.0
        results[f"ILD@{k}"] = float(np.mean(user_ilds)) if user_ilds else 0.0

    return results

class PopularityRecommender:
    """Ranks items globally by frequency in the training set."""
    def __init__(self):
        self.popular_items: List[int] = []

    def fit(self, train_df: pd.DataFrame):
        counts = train_df["movie_id"].value_counts()
        self.popular_items = counts.index.tolist()
        self.item_counts = counts.to_dict()

    def recommend(self, user_history: set, top_k: int = 50) -> List[int]:
        recs = [m for m in self.popular_items if m not in user_history]
        return recs[:top_k]

def run_popularity_baseline(data_dir: Path = DATA_DIR):
    train = pd.read_parquet(data_dir / "train.parquet")
    test = pd.read_parquet(data_dir / "test.parquet")
    movies = pd.read_parquet(data_dir / "movies.parquet")

    model = PopularityRecommender()
    model.fit(train)

    # User train histories
    user_train_history = train.groupby("user_id")["movie_id"].apply(set).to_dict()
    targets = dict(zip(test["user_id"], test["movie_id"]))
    genre_vectors = get_genre_matrix(movies)

    recommendations = {}
    for u in targets:
        hist = user_train_history.get(u, set())
        recommendations[u] = model.recommend(hist, top_k=50)

    metrics = evaluate_recs(
        recommendations=recommendations,
        targets=targets,
        train_counts=model.item_counts,
        total_catalog_size=len(movies),
        genre_vectors=genre_vectors,
        ks=(10, 50),
    )

    print("=== Phase 2: Popularity Baseline Metrics ===")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.4f}")
    return metrics

if __name__ == "__main__":
    run_popularity_baseline()
