import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.baselines import ALL_GENRES, evaluate_recs, get_genre_matrix

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

class WideAndDeepRanker(nn.Module):
    def __init__(self, num_users: int = 6050, num_items: int = 4000, wide_dim: int = 4):
        super().__init__()
        # Wide part
        self.wide_linear = nn.Linear(wide_dim, 1)

        # Deep part
        self.user_emb = nn.Embedding(num_users + 1, 32)
        self.item_emb = nn.Embedding(num_items + 1, 32)
        self.deep_mlp = nn.Sequential(
            nn.Linear(32 + 32 + wide_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, u_ids, i_ids, dense_feats):
        wide_out = self.wide_linear(dense_feats)
        u_e = self.user_emb(u_ids)
        i_e = self.item_emb(i_ids)
        deep_in = torch.cat([u_e, i_e, dense_feats], dim=1)
        deep_out = self.deep_mlp(deep_in)
        return (wide_out + deep_out).squeeze(1)

def extract_features(
    train_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    users_df: pd.DataFrame,
    item_embs: np.ndarray,
    all_movie_ids: np.ndarray,
):
    max_uid = int(users_df["user_id"].max())
    max_mid = int(movies_df["movie_id"].max())

    # 1. Movie genre vectors
    genre_map = {g: i for i, g in enumerate(ALL_GENRES)}
    movie_genres = np.zeros((max_mid + 1, len(ALL_GENRES)), dtype=np.float32)
    recency = np.zeros(max_mid + 1, dtype=np.float32)

    for _, r in movies_df.iterrows():
        mid = int(r["movie_id"])
        for g in str(r["genres"]).split("|"):
            if g in genre_map:
                movie_genres[mid, genre_map[g]] = 1.0
        recency[mid] = (float(r["year"]) - 1920.0) / 85.0

    # 2. Movie popularity
    counts = train_df["movie_id"].value_counts().to_dict()
    max_count = max(counts.values()) if counts else 1
    popularity = np.zeros(max_mid + 1, dtype=np.float32)
    for mid, c in counts.items():
        popularity[mid] = np.log1p(c) / np.log1p(max_count)

    # 3. User genre affinity
    user_genre_profiles = np.zeros((max_uid + 1, len(ALL_GENRES)), dtype=np.float32)
    pos_train = train_df[train_df["rating"] >= 3.0]
    for uid, mid in zip(pos_train["user_id"], pos_train["movie_id"]):
        user_genre_profiles[uid] += movie_genres[mid]
    norms = np.linalg.norm(user_genre_profiles, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    user_genre_profiles = user_genre_profiles / norms

    # Map movie_id to embedding
    mid_to_emb_idx = {int(mid): idx for idx, mid in enumerate(all_movie_ids)}

    return {
        "max_uid": max_uid,
        "max_mid": max_mid,
        "movie_genres": movie_genres,
        "recency": recency,
        "popularity": popularity,
        "user_genre_profiles": user_genre_profiles,
        "item_embs": item_embs,
        "mid_to_emb_idx": mid_to_emb_idx,
    }

def train_and_rank(data_dir: Path = DATA_DIR, artifacts_dir: Path = ARTIFACTS_DIR, epochs: int = 3):
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_parquet(data_dir / "train.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    movies_df = pd.read_parquet(data_dir / "movies.parquet")
    users_df = pd.read_parquet(data_dir / "users.parquet")
    cand_df = pd.read_parquet(artifacts_dir / "retrieval_candidates.parquet")

    item_embs = np.load(artifacts_dir / "item_embeddings.npy")
    all_movie_ids = np.load(artifacts_dir / "item_ids.npy")

    feat_data = extract_features(train_df, movies_df, users_df, item_embs, all_movie_ids)
    user_profiles = feat_data["user_genre_profiles"]
    movie_genres = feat_data["movie_genres"]
    popularity = feat_data["popularity"]
    recency = feat_data["recency"]

    # Build training pairs with sampled negatives
    pos_train = train_df[train_df["rating"] >= 3.5].sample(n=min(200000, len(train_df)), random_state=42)
    user_train_hist = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    all_catalog_ids = all_movie_ids.tolist()
    catalog_set = set(all_catalog_ids)

    pos_u = pos_train["user_id"].values.astype(np.int64)
    pos_i = pos_train["movie_id"].values.astype(np.int64)
    n_pos = len(pos_u)

    # Sample negatives
    neg_u = pos_u.copy()
    neg_i = np.zeros(n_pos, dtype=np.int64)
    for idx, uid in enumerate(neg_u):
        seen = user_train_hist.get(uid, set())
        # Pick random movie not in seen
        cand = np.random.choice(all_catalog_ids)
        while cand in seen:
            cand = np.random.choice(all_catalog_ids)
        neg_i[idx] = cand

    train_u = np.concatenate([pos_u, neg_u])
    train_i = np.concatenate([pos_i, neg_i])
    train_y = np.concatenate([np.ones(n_pos, dtype=np.float32), np.zeros(n_pos, dtype=np.float32)])

    # Compute dense features: [affinity, pop, recency, tower_score_proxy]
    def compute_dense(u_arr, i_arr):
        aff = np.sum(user_profiles[u_arr] * movie_genres[i_arr], axis=1)
        pop = popularity[i_arr]
        rec = recency[i_arr]
        # Approximate score: aff + pop as proxy during training
        comb = aff * 0.6 + pop * 0.4
        return np.column_stack([aff, pop, rec, comb]).astype(np.float32)

    dense_train = compute_dense(train_u, train_i)

    # PyTorch DataLoader
    ds = TensorDataset(
        torch.tensor(train_u, dtype=torch.long),
        torch.tensor(train_i, dtype=torch.long),
        torch.tensor(dense_train, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32),
    )
    dataloader = DataLoader(ds, batch_size=2048, shuffle=True)

    ranker = WideAndDeepRanker(num_users=feat_data["max_uid"], num_items=feat_data["max_mid"], wide_dim=4)
    opt = torch.optim.Adam(ranker.parameters(), lr=3e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    print(f"Training Wide&Deep ranker on {len(train_u):,} samples for {epochs} epochs...")
    ranker.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for b_u, b_i, b_dense, b_y in dataloader:
            opt.zero_grad()
            preds = ranker(b_u, b_i, b_dense)
            loss = criterion(preds, b_y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch}/{epochs} - Loss: {total_loss / len(dataloader):.4f}")

    # Re-rank retrieval candidates
    print("Scoring and re-ranking retrieval candidates...")
    ranker.eval()

    ranked_records = []
    ranked_recs_eval = {}

    with torch.no_grad():
        for _, row in cand_df.iterrows():
            uid = int(row["user_id"])
            c_ids = np.array(row["candidate_ids"], dtype=np.int64)
            tower_scores = np.array(row["candidate_scores"], dtype=np.float32)
            if len(c_ids) == 0:
                ranked_records.append({"user_id": uid, "candidate_ids": [], "candidate_scores": []})
                continue

            u_vec = np.full(len(c_ids), uid, dtype=np.int64)
            aff = np.sum(user_profiles[u_vec] * movie_genres[c_ids], axis=1)
            pop = popularity[c_ids]
            rec = recency[c_ids]
            dense_mat = np.column_stack([aff, pop, rec, tower_scores]).astype(np.float32)

            t_u = torch.tensor(u_vec, dtype=torch.long)
            t_i = torch.tensor(c_ids, dtype=torch.long)
            t_dense = torch.tensor(dense_mat, dtype=torch.float32)

            scores = ranker(t_u, t_i, t_dense).cpu().numpy()
            order = np.argsort(-scores)

            sorted_ids = c_ids[order].tolist()
            sorted_scores = scores[order].tolist()

            ranked_records.append({
                "user_id": uid,
                "candidate_ids": sorted_ids,
                "candidate_scores": sorted_scores,
            })
            ranked_recs_eval[uid] = sorted_ids[:50]

    # Save artifacts
    torch.save(ranker.state_dict(), artifacts_dir / "wide_and_deep.pt")
    ranked_df = pd.DataFrame(ranked_records)
    ranked_df.to_parquet(artifacts_dir / "ranked_candidates.parquet", index=False)

    # Evaluate
    test_targets = dict(zip(test_df["user_id"], test_df["movie_id"]))
    genre_vectors = get_genre_matrix(movies_df)
    train_counts = train_df["movie_id"].value_counts().to_dict()
    metrics = evaluate_recs(
        recommendations=ranked_recs_eval,
        targets=test_targets,
        train_counts=train_counts,
        total_catalog_size=len(movies_df),
        genre_vectors=genre_vectors,
        ks=(10, 50),
    )

    print("=== Phase 4: Wide&Deep Ranker Metrics ===")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.4f}")
    return metrics

if __name__ == "__main__":
    train_and_rank()
