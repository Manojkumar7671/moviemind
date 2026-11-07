"""Honest Content-Based Evaluation on the World Cinema Catalog.

Protocol:
- N = 500 random target films T with fixed seed 42.
- For each target T, sample two other films A and B that share at least one genre AND at least one original language with T.
- Build the profile vector from A and B only.
- Rank the entire catalog excluding A and B.
- A hit is counted if T is in the top 20 recommendations.
- Evaluates:
    (a) Recommender (TF-IDF + prior + MMR lambda=0.7)
    (b) Random ranking baseline
    (c) Popularity ranking baseline (most votes)
- Reports Hit@20 for all three and the lift over both baselines.

Limitations:
Targets share attributes with the liked films by construction, so this is a content-similarity
sanity check and retrieval benchmark, not evidence of personalization quality.
No collaborative filtering metrics (e.g. collaborative Recall/NDCG) apply because IMDb
non-commercial datasets contain no user rating logs.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.global_recommender import GlobalContentRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("global_eval")

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DOCS_DIR = ROOT_DIR / "docs"


def run_global_evaluation(
    n_samples: int = 500,
    seed: int = 42,
    artifacts_dir: Path = ARTIFACTS_DIR,
    docs_dir: Path = DOCS_DIR,
) -> Dict[str, Any]:
    catalog_path = artifacts_dir / "global_catalog.parquet"
    rec = GlobalContentRecommender(catalog_path)
    df = rec.df
    n_items = len(df)
    rng = np.random.RandomState(seed)

    logger.info(f"Running honest global evaluation (N={n_samples}, seed={seed}) over {n_items:,} catalog items...")

    # 1. Build inverted indexes for fast candidate pair discovery
    genre_to_indices = defaultdict(set)
    lang_to_indices = defaultdict(set)

    for idx, row in df.iterrows():
        for g in row["genres"].split("|"):
            if g.strip():
                genre_to_indices[g.strip().lower()].add(idx)
        for l in row["original_languages"]:
            if str(l).strip():
                lang_to_indices[str(l).strip().lower()].add(idx)

    # 2. Precompute global popularity list (sorted by votes descending)
    pop_sorted_ids = df.sort_values(by="votes", ascending=False)["id"].tolist()
    all_ids = df["id"].tolist()

    # 3. Find valid target candidates (must have >= 2 items sharing >= 1 genre and >= 1 language)
    valid_targets = []
    for idx in range(n_items):
        row = df.iloc[idx]
        gs = [g.strip().lower() for g in row["genres"].split("|") if g.strip()]
        ls = [str(l).strip().lower() for l in row["original_languages"] if str(l).strip()]
        if not gs or not ls:
            continue
        g_matches = set.union(*(genre_to_indices[g] for g in gs))
        l_matches = set.union(*(lang_to_indices[l] for l in ls))
        cands = (g_matches & l_matches) - {idx}
        if len(cands) >= 2:
            valid_targets.append(idx)

    logger.info(f"Found {len(valid_targets):,} eligible target films with >=2 matching candidates.")
    target_sample = rng.choice(valid_targets, size=min(n_samples, len(valid_targets)), replace=False)
    actual_n = len(target_sample)

    hits_rec = 0
    hits_rand = 0
    hits_pop = 0

    # 4. Evaluation loop
    for step, t_idx in enumerate(target_sample):
        row = df.iloc[t_idx]
        t_id = str(row["id"])
        gs = [g.strip().lower() for g in row["genres"].split("|") if g.strip()]
        ls = [str(l).strip().lower() for l in row["original_languages"] if str(l).strip()]
        g_matches = set.union(*(genre_to_indices[g] for g in gs))
        l_matches = set.union(*(lang_to_indices[l] for l in ls))
        cands = sorted(list((g_matches & l_matches) - {t_idx}))

        # Sample two distinct condition films A and B
        ab_indices = rng.choice(cands, size=2, replace=False)
        a_id = rec.idx_to_id[ab_indices[0]]
        b_id = rec.idx_to_id[ab_indices[1]]
        liked_ids = [a_id, b_id]

        # (a) Recommender ranking (excluding A and B)
        recs = rec.recommend(liked_ids=liked_ids, n=20, lmbda=0.7)
        rec_ids = [r["movie_id"] for r in recs]
        if t_id in rec_ids:
            hits_rec += 1

        # (b) Random ranking baseline (sample 20 random items excluding A and B)
        avail = [i for i in all_ids if i not in liked_ids]
        rand_20 = rng.choice(avail, size=20, replace=False)
        if t_id in rand_20:
            hits_rand += 1

        # (c) Popularity ranking baseline (top 20 most votes excluding A and B)
        pop_20 = [pid for pid in pop_sorted_ids if pid not in liked_ids][:20]
        if t_id in pop_20:
            hits_pop += 1

        if (step + 1) % 100 == 0 or (step + 1) == actual_n:
            logger.info(f"Evaluated {step + 1}/{actual_n} targets. Hits: Rec={hits_rec}, Rand={hits_rand}, Pop={hits_pop}")

    rec_hit_rate = round(hits_rec / actual_n, 4)
    rand_hit_rate = round(hits_rand / actual_n, 4)
    pop_hit_rate = round(hits_pop / actual_n, 4)

    lift_over_random = round(rec_hit_rate - rand_hit_rate, 4)
    lift_over_popularity = round(rec_hit_rate - pop_hit_rate, 4)

    results = {
        "n_samples": actual_n,
        "seed": seed,
        "catalog_size": n_items,
        "recommender_hit20": rec_hit_rate,
        "recommender_hits": hits_rec,
        "random_hit20": rand_hit_rate,
        "random_hits": hits_rand,
        "popularity_hit20": pop_hit_rate,
        "popularity_hits": hits_pop,
        "lift_over_random": lift_over_random,
        "lift_over_popularity": lift_over_popularity,
        "honesty_disclaimer": (
            "Evaluated strictly via content-based target recovery where targets share genres and languages "
            "with conditioned liked items. No collaborative filtering performance (collaborative Recall/NDCG) "
            "is claimed because IMDb datasets contain no individual user rating logs."
        ),
        "limitations": (
            "Target films share attributes with the liked films by construction, so this is a content-similarity "
            "sanity check and candidate-retrieval benchmark, not evidence of personalization quality."
        ),
    }

    # Save artifacts/global_eval_results.json
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_path = artifacts_dir / "global_eval_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved evaluation results to {results_path}")

    # Generate docs/global_eval.md
    docs_dir.mkdir(parents=True, exist_ok=True)
    doc_path = docs_dir / "global_eval.md"
    md_content = f"""# World Cinema Recommender Evaluation

**Evaluation Benchmark**: Honest Content Target Recovery ($N={actual_n}$, seed={seed}) over {n_items:,} films from IMDb non-commercial datasets with Wikidata (CC0) origin and language metadata.

## Protocol
For each target film $T$:
1. Identify all other films in the catalog sharing at least one genre and at least one original language with $T$.
2. Randomly sample two distinct conditioning films $A$ and $B$.
3. Construct the user taste profile from $A$ and $B$ only.
4. Rank the entire catalog excluding $A$ and $B$.
5. A **Hit** is scored if target film $T$ appears in the top-20 recommendations ($K=20$).

## Results Comparison

| Model / Pipeline | Hit@20 | Hits ({actual_n} targets) | Absolute Lift vs Baseline |
| :--- | :---: | :---: | :---: |
| **MovieMind Content Recommender (TF-IDF + Prior + MMR $\lambda=0.7$)** | **{rec_hit_rate * 100:.2f}%** | **{hits_rec}** | — |
| **Random Ranking Baseline** | {rand_hit_rate * 100:.2f}% | {hits_rand} | {lift_over_random * 100:+.2f}% |
| **Popularity Ranking Baseline (Most Votes)** | {pop_hit_rate * 100:.2f}% | {hits_pop} | {lift_over_popularity * 100:+.2f}% |

- **Lift over Random**: **{lift_over_random * 100:+.2f}%** ({rec_hit_rate * 100:.2f}% vs {rand_hit_rate * 100:.2f}%)
- **Lift over Popularity**: **{lift_over_popularity * 100:+.2f}%** ({rec_hit_rate * 100:.2f}% vs {pop_hit_rate * 100:.2f}%)

## Honest Limitations
1. **Sanity Check Only**: The target film shares attributes (genre and original language) with the conditioned liked films by construction. This evaluation is an attribute-conditioned retrieval sanity check, not proof of individual user satisfaction or personalization quality.
2. **No Collaborative Filtering**: IMDb non-commercial datasets do not publish user-level viewing or rating histories. No collaborative filtering performance (collaborative Recall@K, NDCG@K) is claimed on this catalog.
3. **Sparse & Multi-Origin Films**: Films with sparse metadata or co-productions with multiple countries are matched across all listed countries.
"""
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Generated evaluation report at {doc_path}")

    print("\n=== World Cinema Evaluation Results ===")
    print(f"Targets evaluated (N):             {actual_n} (seed={seed})")
    print(f"Recommender Hit@20:                {hits_rec}/{actual_n} ({rec_hit_rate * 100:.2f}%)")
    print(f"Random Ranking Hit@20:             {hits_rand}/{actual_n} ({rand_hit_rate * 100:.2f}%)")
    print(f"Popularity Ranking Hit@20:         {hits_pop}/{actual_n} ({pop_hit_rate * 100:.2f}%)")
    print(f"Lift over Random Baseline:         {lift_over_random * 100:+.2f}%")
    print(f"Lift over Popularity Baseline:     {lift_over_popularity * 100:+.2f}%")
    print("=======================================\n")

    return results


if __name__ == "__main__":
    run_global_evaluation()
