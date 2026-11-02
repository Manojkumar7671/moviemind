import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from pathlib import Path
from typing import Dict, List, Tuple
import faiss
faiss.omp_set_num_threads(1)
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.baselines import ALL_GENRES, evaluate_recs, get_genre_matrix

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
EMBED_DIM = 64
TEMPERATURE = 0.07
AGE_MAP = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}

class UserTower(nn.Module):
    def __init__(self, num_users: int = 6050, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.user_emb = nn.Embedding(num_users + 1, 32)
        self.gender_emb = nn.Embedding(2, 4)
        self.age_emb = nn.Embedding(7, 8)
        self.occ_emb = nn.Embedding(22, 8)
        self.fc = nn.Sequential(
            nn.Linear(32 + 4 + 8 + 8 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )

    def forward(self, u_id, gender, age, occ, hist_avg):
        u = self.user_emb(u_id)
        g = self.gender_emb(gender)
        a = self.age_emb(age)
        o = self.occ_emb(occ)
        h = hist_avg.unsqueeze(1)
        x = torch.cat([u, g, a, o, h], dim=1)
        return F.normalize(self.fc(x), p=2, dim=1)

class ItemTower(nn.Module):
    def __init__(self, num_items: int = 4000, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, 32)
        self.genre_fc = nn.Linear(len(ALL_GENRES), 16)
        self.fc = nn.Sequential(
            nn.Linear(32 + 16 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )

    def forward(self, i_id, genres, year_norm):
        it = self.item_emb(i_id)
        gf = self.genre_fc(genres)
        yr = year_norm.unsqueeze(1)
        x = torch.cat([it, gf, yr], dim=1)
        return F.normalize(self.fc(x), p=2, dim=1)

class TwoTowerModel(nn.Module):
    def __init__(self, num_users: int = 6050, num_items: int = 4000, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.user_tower = UserTower(num_users, embed_dim)
        self.item_tower = ItemTower(num_items, embed_dim)

    def forward(self, u_tensors, i_tensors):
        u_emb = self.user_tower(*u_tensors)
        i_emb = self.item_tower(*i_tensors)
        return u_emb, i_emb

def build_feature_tables(train_df: pd.DataFrame, users_df: pd.DataFrame, movies_df: pd.DataFrame):
    max_uid = int(users_df["user_id"].max())
    max_mid = int(movies_df["movie_id"].max())
    user_avg = train_df.groupby("user_id")["rating"].mean().to_dict()

    genders = torch.zeros(max_uid + 1, dtype=torch.long)
    ages = torch.zeros(max_uid + 1, dtype=torch.long)
    occs = torch.zeros(max_uid + 1, dtype=torch.long)
    h_avgs = torch.full((max_uid + 1,), 0.7, dtype=torch.float32)

    for _, r in users_df.iterrows():
        uid = int(r["user_id"])
        genders[uid] = int(r["gender_num"])
        ages[uid] = AGE_MAP.get(int(r["age"]), 2)
        occs[uid] = min(int(r["occupation"]), 20)
        h_avgs[uid] = float(user_avg.get(uid, 3.5)) / 5.0

    genre_map = {g: i for i, g in enumerate(ALL_GENRES)}
    genres_table = torch.zeros((max_mid + 1, len(ALL_GENRES)), dtype=torch.float32)
    years_table = torch.zeros(max_mid + 1, dtype=torch.float32)

    for _, r in movies_df.iterrows():
        mid = int(r["movie_id"])
        for g in str(r["genres"]).split("|"):
            if g in genre_map:
                genres_table[mid, genre_map[g]] = 1.0
        years_table[mid] = (float(r["year"]) - 1990.0) / 15.0

    user_tables = (genders, ages, occs, h_avgs)
    item_tables = (genres_table, years_table)
    return user_tables, item_tables

def train_and_export_retrieval(data_dir: Path = DATA_DIR, artifacts_dir: Path = ARTIFACTS_DIR, epochs: int = 4):
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_parquet(data_dir / "train.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    movies_df = pd.read_parquet(data_dir / "movies.parquet")
    users_df = pd.read_parquet(data_dir / "users.parquet")

    user_tables, item_tables = build_feature_tables(train_df, users_df, movies_df)
    genders, ages, occs, h_avgs = user_tables
    genres_table, years_table = item_tables

    pos_train = train_df[train_df["rating"] >= 3.0]
    u_tensor = torch.tensor(pos_train["user_id"].values, dtype=torch.long)
    i_tensor = torch.tensor(pos_train["movie_id"].values, dtype=torch.long)

    dataset = TensorDataset(u_tensor, i_tensor)
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True)

    max_user = int(users_df["user_id"].max())
    max_item = int(movies_df["movie_id"].max())
    model = TwoTowerModel(num_users=max_user, num_items=max_item, embed_dim=EMBED_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Training Two-Tower model on {len(pos_train):,} interactions for {epochs} epochs...")
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_u, batch_i in dataloader:
            optimizer.zero_grad()
            u_tensors = (batch_u, genders[batch_u], ages[batch_u], occs[batch_u], h_avgs[batch_u])
            i_tensors = (batch_i, genres_table[batch_i], years_table[batch_i])
            u_emb, i_emb = model(u_tensors, i_tensors)
            logits = torch.matmul(u_emb, i_emb.T) / TEMPERATURE
            loss = loss_fn(logits, torch.arange(len(logits)))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch}/{epochs} - Loss: {total_loss / len(dataloader):.4f}")

    # Compute item embeddings for entire catalog
    model.eval()
    all_movie_ids = movies_df["movie_id"].values.astype(int)
    all_i_tensor = torch.tensor(all_movie_ids, dtype=torch.long)
    with torch.no_grad():
        cat_i_tensors = (all_i_tensor, genres_table[all_i_tensor], years_table[all_i_tensor])
        item_embs = model.item_tower(*cat_i_tensors).cpu().numpy().astype(np.float32)

    faiss.normalize_L2(item_embs)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(item_embs)

    # Save artifacts
    torch.save(model.state_dict(), artifacts_dir / "two_tower.pt")
    np.save(artifacts_dir / "item_embeddings.npy", item_embs)
    np.save(artifacts_dir / "item_ids.npy", all_movie_ids)
    faiss.write_index(index, str(artifacts_dir / "faiss_index.bin"))

    # Generate Top-200 candidates per test user
    test_targets = dict(zip(test_df["user_id"], test_df["movie_id"]))
    user_train_hist = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()
    eval_uids = sorted(list(test_targets.keys()))
    eval_u_tensor = torch.tensor(eval_uids, dtype=torch.long)

    with torch.no_grad():
        eval_u_tensors = (
            eval_u_tensor,
            genders[eval_u_tensor],
            ages[eval_u_tensor],
            occs[eval_u_tensor],
            h_avgs[eval_u_tensor],
        )
        user_embs = model.user_tower(*eval_u_tensors).cpu().numpy().astype(np.float32)

    faiss.normalize_L2(user_embs)
    scores, indices = index.search(user_embs, 250)

    candidate_records = []
    eval_recs = {}
    for idx, uid in enumerate(eval_uids):
        seen = user_train_hist.get(uid, set())
        u_cands = []
        u_scores = []
        for rank, movie_idx in enumerate(indices[idx]):
            mid = int(all_movie_ids[movie_idx])
            if mid not in seen:
                u_cands.append(mid)
                u_scores.append(float(scores[idx][rank]))
            if len(u_cands) >= 200:
                break
        candidate_records.append({
            "user_id": uid,
            "candidate_ids": u_cands,
            "candidate_scores": u_scores,
        })
        eval_recs[uid] = u_cands[:50]

    cand_df = pd.DataFrame(candidate_records)
    cand_df.to_parquet(artifacts_dir / "retrieval_candidates.parquet", index=False)

    genre_vectors = get_genre_matrix(movies_df)
    train_counts = train_df["movie_id"].value_counts().to_dict()
    metrics = evaluate_recs(
        recommendations=eval_recs,
        targets=test_targets,
        train_counts=train_counts,
        total_catalog_size=len(movies_df),
        genre_vectors=genre_vectors,
        ks=(10, 50),
    )

    print("=== Phase 3: Two-Tower Retrieval Metrics ===")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.4f}")
    return metrics

if __name__ == "__main__":
    train_and_export_retrieval()
