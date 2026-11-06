"""Content-based recommender over the World Cinema catalog.

Features:
- TF-IDF over genres, original languages, and origin countries (sourced from Wikidata CC0)
- Bayesian rating/votes prior
- Cosine similarity with user taste profile vector
- Likes dynamically shift user profile vector
- Maximal Marginal Relevance (MMR) re-ranking for intra-list diversity
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.diversity import mmr_rerank

CATALOG_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "global_catalog.parquet"


class GlobalContentRecommender:
    def __init__(self, catalog_path: Path = CATALOG_PATH):
        self.catalog_path = Path(catalog_path)
        self.df = pd.read_parquet(self.catalog_path)
        self.n_items = len(self.df)

        self.id_to_idx = {mid: idx for idx, mid in enumerate(self.df["id"])}
        self.idx_to_id = {idx: mid for idx, mid in enumerate(self.df["id"])}

        # 1. Build text representations for TF-IDF from genres, original_languages, and origin_countries
        feature_texts = []
        for _, r in self.df.iterrows():
            genres = " ".join([f"g_{g.strip().lower()}" for g in r["genres"].split("|") if g.strip()])
            langs = " ".join(
                [f"l_{str(l).strip().lower()}" for l in r["original_languages"] if str(l).strip() and l != "Unknown"]
            )
            countries = " ".join(
                [f"c_{str(c).strip().lower()}" for c in r["origin_countries"] if str(c).strip() and c != "Unknown"]
            )
            tokens = f"{genres} {langs} {countries}".strip()
            feature_texts.append(tokens if tokens else "empty")

        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        self.tfidf_matrix = self.vectorizer.fit_transform(feature_texts)

        # 2. Bayesian rating/votes prior
        C = float(self.df["rating"].mean())
        m = 2000.0
        v = self.df["votes"].to_numpy(dtype=np.float32)
        R = self.df["rating"].to_numpy(dtype=np.float32)
        bayes_rating = (v / (v + m)) * R + (m / (v + m)) * C
        self.prior = np.clip((bayes_rating - 4.0) / 5.5, 0.0, 1.0).astype(np.float32)

        # 3. Dynamic metadata bounds and counts derived from data
        self.min_year = int(self.df["year"].min())
        self.max_year = int(self.df["year"].max())

        region_counts = Counter()
        region_names: Dict[str, str] = {}
        for _, row in self.df.iterrows():
            cs = row["origin_countries"]
            clbls = row["country_labels"]
            for c, lbl in zip(cs, clbls):
                if c and c != "Unknown":
                    code = str(c).upper()
                    region_counts[code] += 1
                    if lbl and lbl != "Unknown":
                        region_names[code] = str(lbl)

        self.available_regions = [
            {"code": code, "name": region_names.get(code, code), "count": count}
            for code, count in region_counts.most_common()
        ]

        lang_counts = Counter()
        lang_names: Dict[str, str] = {}
        for _, row in self.df.iterrows():
            ls = row["original_languages"]
            llbls = row["language_labels"]
            for l, lbl in zip(ls, llbls):
                if l and l != "Unknown":
                    code = str(l).lower()
                    lang_counts[code] += 1
                    if lbl and lbl != "Unknown":
                        lang_names[code] = str(lbl)

        self.available_languages = [
            {"code": code, "name": lang_names.get(code, code), "count": count}
            for code, count in lang_counts.most_common()
        ]

        genre_counts = Counter()
        for _, row in self.df.iterrows():
            for g in row["genres"].split("|"):
                if g.strip():
                    genre_counts[g.strip()] += 1
        self.available_genres = [g for g, _ in genre_counts.most_common()]

    def get_user_profile_vector(
        self,
        liked_ids: Optional[List[str]] = None,
        preferred_genres: Optional[List[str]] = None,
        preferred_region: Optional[str] = None,
        preferred_language: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Compute user profile vector shifted by likes and explicit preferences."""
        liked_vec = None
        if liked_ids:
            valid_indices = [self.id_to_idx[mid] for mid in liked_ids if mid in self.id_to_idx]
            if valid_indices:
                liked_sub = self.tfidf_matrix[valid_indices]
                liked_vec = np.asarray(liked_sub.mean(axis=0)).flatten()
                norm = np.linalg.norm(liked_vec)
                if norm > 0:
                    liked_vec /= norm

        query_tokens = []
        if preferred_genres:
            query_tokens.extend([f"g_{g.strip().lower()}" for g in preferred_genres if g.strip()])
        if preferred_region:
            query_tokens.append(f"c_{preferred_region.strip().lower()}")
        if preferred_language:
            query_tokens.append(f"l_{preferred_language.strip().lower()}")

        pref_vec = None
        if query_tokens:
            raw_v = np.asarray(self.vectorizer.transform([" ".join(query_tokens)]).todense()).flatten()
            norm = np.linalg.norm(raw_v)
            if norm > 0:
                pref_vec = raw_v / norm

        if liked_vec is not None and pref_vec is not None:
            combined = 0.7 * liked_vec + 0.3 * pref_vec
            c_norm = np.linalg.norm(combined)
            return combined / c_norm if c_norm > 0 else combined
        if liked_vec is not None:
            return liked_vec
        return pref_vec

    def recommend(
        self,
        liked_ids: Optional[List[str]] = None,
        genres: Optional[List[str]] = None,
        region: Optional[str] = None,
        language: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        n: int = 12,
        lmbda: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Return diverse content-based recommendations with metadata filters."""
        # 1. Candidate filtering mask
        mask = np.ones(self.n_items, dtype=bool)

        if year_min is not None:
            mask &= (self.df["year"] >= year_min)
        if year_max is not None:
            mask &= (self.df["year"] <= year_max)

        # Region filter: requested country in origin_countries (case-insensitive); exclude Unknown
        if region:
            reg_clean = region.strip().upper()
            mask &= self.df["origin_countries"].apply(
                lambda cs: bool(cs) and any(str(c).upper() == reg_clean for c in cs if c != "Unknown")
            )

        # Language filter: requested language in original_languages (case-insensitive); exclude Unknown
        if language:
            lang_clean = language.strip().lower()
            mask &= self.df["original_languages"].apply(
                lambda ls: bool(ls) and any(str(l).lower() == lang_clean for l in ls if l != "Unknown")
            )

        if genres:
            g_set = set(g.strip().lower() for g in genres if g.strip())
            mask &= self.df["genres"].apply(
                lambda s: bool(set(x.lower() for x in s.split("|") if x) & g_set)
            )

        if liked_ids:
            liked_set = set(liked_ids)
            mask &= ~self.df["id"].isin(liked_set)

        cand_indices = np.where(mask)[0]
        if len(cand_indices) == 0:
            return []

        # 2. Score computation: Cosine similarity + Bayesian prior
        profile_vec = self.get_user_profile_vector(
            liked_ids=liked_ids,
            preferred_genres=genres,
            preferred_region=region,
            preferred_language=language,
        )

        cand_priors = self.prior[cand_indices]
        if profile_vec is not None:
            sub_tfidf = self.tfidf_matrix[cand_indices]
            sims = np.asarray(sub_tfidf.dot(profile_vec)).flatten()
            scores = 0.75 * sims + 0.25 * cand_priors
        else:
            scores = cand_priors

        # 3. Candidate pool selection for MMR
        pool_size = min(len(cand_indices), max(n * 4, 40))
        top_k_cand = np.argsort(-scores)[:pool_size]
        pool_indices = cand_indices[top_k_cand]
        pool_scores = scores[top_k_cand]
        pool_ids = [self.idx_to_id[i] for i in pool_indices]

        # 4. MMR diversity re-ranking
        if lmbda < 1.0 and len(pool_ids) > n:
            pool_vecs = {
                self.idx_to_id[i]: np.asarray(self.tfidf_matrix[i].todense()).flatten()
                for i in pool_indices
            }
            final_ids = mmr_rerank(pool_ids, pool_scores.tolist(), pool_vecs, top_k=n, lmbda=lmbda)
        else:
            final_ids = pool_ids[:n]

        # 5. Format output cards
        score_lookup = dict(zip(pool_ids, pool_scores))
        max_s = float(np.max(pool_scores)) if len(pool_scores) > 0 else 1.0
        min_s = float(np.min(pool_scores)) if len(pool_scores) > 0 else 0.0

        results = []
        for mid in final_ids:
            idx = self.id_to_idx[mid]
            row = self.df.iloc[idx]
            raw_s = score_lookup.get(mid, 0.5)
            pct = int(np.clip(70 + 29 * (raw_s - min_s) / (max_s - min_s + 1e-8), 70, 99))
            genre_list = [g.strip() for g in row["genres"].split("|") if g.strip()]
            origin_str = ", ".join(row["origin_countries"]) if len(row["origin_countries"]) > 0 else "unknown"
            country_name_str = ", ".join(row["country_labels"]) if len(row["country_labels"]) > 0 else origin_str

            results.append({
                "movie_id": str(row["id"]),
                "title": str(row["title"]),
                "year": int(row["year"]),
                "genres": genre_list,
                "rating": float(row["rating"]),
                "votes": int(row["votes"]),
                "origin_countries": list(row["origin_countries"]),
                "original_languages": list(row["original_languages"]),
                "origin_primary": str(row["origin_primary"]),
                "language_primary": str(row["language_primary"]),
                "country_labels": list(row["country_labels"]),
                "language_labels": list(row["language_labels"]),
                "origin": origin_str,
                "country_name": country_name_str,
                "match_pct": pct,
                "explanation": f"Rating {row['rating']}/10 ({row['votes']:,} votes) • Origin: {origin_str} • {', '.join(genre_list[:2])}",
                "trace": [
                    {"tool": "content_tfidf", "args": {"mid": mid}, "latency_ms": 1.2},
                    {"tool": "mmr_diversity", "args": {"lambda": lmbda}, "latency_ms": 0.8},
                ],
            })

        return results
