import json
import os
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
from typing import Dict, List, Optional
import faiss
faiss.omp_set_num_threads(1)
import numpy as np
import pandas as pd

from src.baselines import ALL_GENRES
from src.utils import clean_title, parse_genres

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

class ColdStartRecommender:
    """Recommends movies for new users picking 3-5 favourite genres."""
    def __init__(self, data_dir: Path = DATA_DIR, artifacts_dir: Path = ARTIFACTS_DIR):
        self.movies_df = pd.read_parquet(data_dir / "movies.parquet")
        self.item_embs = np.load(artifacts_dir / "item_embeddings.npy", mmap_mode="r")
        self.item_ids = np.load(artifacts_dir / "item_ids.npy")
        try:
            self.index = faiss.read_index(str(artifacts_dir / "faiss_index.bin"), faiss.IO_FLAG_MMAP)
        except Exception:
            self.index = faiss.read_index(str(artifacts_dir / "faiss_index.bin"))

        # Precompute movie metadata map
        self.movie_meta = self.movies_df.set_index("movie_id").to_dict(orient="index")
        self.mid_to_idx = {int(mid): idx for idx, mid in enumerate(self.item_ids)}

        # Popularity per movie from precomputed json or train
        log_pop_file = artifacts_dir / "log_pop.json"
        if log_pop_file.exists():
            with open(log_pop_file, "r") as f:
                raw_pop = json.load(f)
            self.log_pop = {int(k): float(v) for k, v in raw_pop.items()}
        else:
            train_df = pd.read_parquet(data_dir / "train.parquet")
            counts = train_df["movie_id"].value_counts().to_dict()
            max_c = max(counts.values()) if counts else 1
            self.log_pop = {int(mid): float(np.log1p(counts.get(mid, 0)) / np.log1p(max_c)) for mid in self.item_ids}

    def build_profile_vector(self, selected_genres: List[str]) -> np.ndarray:
        """Construct user profile vector as mean embedding of top movies in selected genres."""
        genre_set = set(g.strip().lower() for g in selected_genres)
        matching_indices = []

        for mid, meta in self.movie_meta.items():
            item_genres = set(g.strip().lower() for g in str(meta["genres"]).split("|"))
            if genre_set.intersection(item_genres) and mid in self.mid_to_idx:
                matching_indices.append(self.mid_to_idx[mid])

        if matching_indices:
            vecs = self.item_embs[matching_indices]
            profile_vec = np.mean(vecs, axis=0, keepdims=True).astype(np.float32)
        else:
            profile_vec = np.mean(self.item_embs, axis=0, keepdims=True).astype(np.float32)

        faiss.normalize_L2(profile_vec)
        return profile_vec

    def recommend(
        self,
        selected_genres: List[str],
        top_n: int = 10,
        alpha: float = 0.5,
    ) -> List[Dict]:
        """Hybrid cold-start: item embedding similarity + popularity within selected genres."""
        if not selected_genres:
            selected_genres = ["Drama", "Comedy", "Action"]

        profile_vec = self.build_profile_vector(selected_genres)
        # 1. Embedding similarity across all catalog movies
        emb_sims = np.dot(self.item_embs, profile_vec.T).squeeze(1)

        # 2. Genre popularity score
        genre_set = set(g.strip().lower() for g in selected_genres)
        scores = []

        for idx, mid in enumerate(self.item_ids):
            meta = self.movie_meta.get(mid, {})
            item_genres = set(g.strip().lower() for g in str(meta.get("genres", "")).split("|"))
            overlap = len(genre_set.intersection(item_genres))
            genre_boost = overlap / max(len(genre_set), 1)

            pop = self.log_pop.get(mid, 0.0)
            genre_pop_score = 0.7 * genre_boost + 0.3 * pop
            emb_score = float(emb_sims[idx])

            total_score = alpha * emb_score + (1.0 - alpha) * genre_pop_score
            scores.append((mid, total_score, emb_score, genre_pop_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_items = scores[:top_n]

        recs = []
        for mid, score, emb_s, gp_s in top_items:
            m_info = self.movie_meta[mid]
            recs.append({
                "movie_id": int(mid),
                "title": clean_title(m_info["title"]),
                "genres": parse_genres(m_info["genres"]),
                "year": int(m_info["year"]),
                "score": round(float(score), 4),
            })
        return recs

def run_coldstart_demo():
    print("=== Phase 6: Cold-Start Onboarding Recommendations ===")
    recommender = ColdStartRecommender()

    test_profiles = [
        ["Action", "Sci-Fi", "Thriller"],
        ["Animation", "Children's", "Fantasy"],
        ["Crime", "Drama", "Film-Noir"],
    ]

    for genres in test_profiles:
        print(f"\nUser Picked Genres: {genres}")
        recs = recommender.recommend(genres, top_n=5)
        for rank, r in enumerate(recs, 1):
            print(f"  {rank}. {r['title']} ({r['year']}) - [{r['genres']}] (score: {r['score']})")

if __name__ == "__main__":
    run_coldstart_demo()
