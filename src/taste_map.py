import os
import pickle
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.utils import clean_title, parse_genres

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

class TasteMapService:
    """Precomputed 2D projection of item embeddings and real-time user taste projection."""
    def __init__(self, data_dir: Path = DATA_DIR, artifacts_dir: Path = ARTIFACTS_DIR, agent: Optional[Any] = None):
        self.item_ids = np.load(artifacts_dir / "item_ids.npy")
        self.item_embs = np.load(artifacts_dir / "item_embeddings.npy", mmap_mode="r")
        self.coords_2d = np.load(artifacts_dir / "taste_map_coords.npy")

        with open(artifacts_dir / "pca_model.pkl", "rb") as f:
            self.pca = pickle.load(f)

        self.mid_to_idx = {int(mid): idx for idx, mid in enumerate(self.item_ids)}
        if agent is not None and hasattr(agent, "movie_meta"):
            self.movie_meta = agent.movie_meta
            self.movies_df = agent.movies_df
        else:
            self.movies_df = pd.read_parquet(data_dir / "movies.parquet")
            self.movie_meta = self.movies_df.set_index("movie_id").to_dict(orient="index")

        # Bounds for normalization from precomputed 2D coords
        self.c_min = self.coords_2d.min(axis=0)
        self.c_max = self.coords_2d.max(axis=0)

    def normalize_coord(self, raw_coord: np.ndarray) -> np.ndarray:
        norm = 2.0 * (raw_coord - self.c_min) / (self.c_max - self.c_min + 1e-8) - 1.0
        return np.clip(norm, -1.0, 1.0)

    def get_taste_map(
        self,
        liked_ids: List[int],
        fallback_genres: Optional[List[str]] = None,
        max_points: int = 1200,
    ) -> Dict[str, Any]:
        # Sample points across genres for clean visualization
        n_items = len(self.item_ids)
        step = max(1, n_items // max_points)
        sampled_indices = list(range(0, n_items, step))

        # Always include liked items in points
        liked_set = set(liked_ids)
        for mid in liked_ids:
            if mid in self.mid_to_idx:
                idx = self.mid_to_idx[mid]
                if idx not in sampled_indices:
                    sampled_indices.append(idx)

        points = []
        for idx in sampled_indices:
            mid = int(self.item_ids[idx])
            meta = self.movie_meta.get(mid, {})
            genres = parse_genres(meta.get("genres", "Other"))
            primary_genre = genres[0] if genres else "Other"
            cx, cy = self.coords_2d[idx]
            points.append({
                "id": mid,
                "title": clean_title(meta.get("title", f"Movie {mid}")),
                "year": int(meta.get("year", 1990)),
                "x": round(float(cx), 3),
                "y": round(float(cy), 3),
                "genre": primary_genre,
                "genres": genres,
                "is_liked": bool(mid in liked_set),
            })

        # Calculate user projection point
        if liked_ids:
            valid_vecs = [self.item_embs[self.mid_to_idx[m]] for m in liked_ids if m in self.mid_to_idx]
            if valid_vecs:
                u_vec = np.mean(valid_vecs, axis=0, keepdims=True)
            else:
                u_vec = np.zeros((1, 64), dtype=np.float32)
        else:
            # Cold-start profile from genres
            g_set = set(g.lower() for g in (fallback_genres or ["action", "drama"]))
            matching = [
                self.mid_to_idx[mid]
                for mid, m in self.movie_meta.items()
                if mid in self.mid_to_idx and g_set.intersection(set(g.lower() for g in parse_genres(m["genres"])))
            ]
            if matching:
                u_vec = np.mean(self.item_embs[matching[:50]], axis=0, keepdims=True)
            else:
                u_vec = np.mean(self.item_embs[:50], axis=0, keepdims=True)

        raw_u_coord = self.pca.transform(u_vec)[0]
        norm_u = self.normalize_coord(raw_u_coord)

        return {
            "points": points,
            "user_point": {
                "x": round(float(norm_u[0]), 3),
                "y": round(float(norm_u[1]), 3),
                "label": "Your Taste Center",
            },
            "liked_ids": liked_ids,
        }
