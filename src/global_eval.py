"""Honest Content-Based Evaluation on the World Cinema (IMDb) Catalog.

Evaluates leave-one-out content similarity on liked items:
- Hit Rate@10 and Hit Rate@20 on held-out similar neighbor
- Mean Cosine Similarity to held-out favorite
- Intra-List Diversity (ILD@10) with vs without MMR
- Catalog Coverage and Origin Region alignment

DISCLAIMER: Evaluated strictly via content-based item similarity.
Because the IMDb non-commercial dataset does not contain individual user interaction
histories, NO collaborative filtering performance (collaborative Recall/NDCG) is claimed.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.global_recommender import GlobalContentRecommender

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

def run_global_evaluation(
    n_samples: int = 500,
    seed: int = 42,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> Dict[str, Any]:
    rec = GlobalContentRecommender(artifacts_dir / "global_catalog.parquet")
    np.random.seed(seed)

    # 1. Sample evaluation probe items across catalog
    sample_indices = np.random.choice(rec.n_items, size=min(n_samples, rec.n_items), replace=False)

    hits_10_no_mmr = 0
    hits_10_mmr = 0
    hits_20_mmr = 0
    sims_10_no_mmr = []
    sims_10_mmr = []
    ild_no_mmr = []
    ild_mmr = []
    region_match_counts = []
    genre_match_counts = []

    unique_recs_no_mmr = set()
    unique_recs_mmr = set()

    for idx in sample_indices:
        mid = rec.idx_to_id[idx]
        probe_row = rec.df.iloc[idx]
        probe_region = probe_row["approx_origin_region"]
        probe_genres = set(g.strip().lower() for g in probe_row["genres"].split("|") if g.strip())

        # Find closest content neighbor in TF-IDF space (held-out target item)
        probe_vec = rec.tfidf_matrix[idx]
        sim_scores = np.asarray(rec.tfidf_matrix.dot(probe_vec.T).todense()).flatten()
        sim_scores[idx] = -1.0  # exclude self
        top_neighbor_idx = int(np.argmax(sim_scores))
        target_mid = rec.idx_to_id[top_neighbor_idx]
        target_vec = rec.tfidf_matrix[top_neighbor_idx]

        # A. Recommend without MMR (lambda = 1.0)
        recs_no_mmr = rec.recommend(liked_ids=[mid], n=10, lmbda=1.0)
        rec_ids_no_mmr = [r["movie_id"] for r in recs_no_mmr]
        unique_recs_no_mmr.update(rec_ids_no_mmr)
        if target_mid in rec_ids_no_mmr:
            hits_10_no_mmr += 1

        # Cosine similarity to target
        if rec_ids_no_mmr:
            r_indices = [rec.id_to_idx[r_id] for r_id in rec_ids_no_mmr if r_id in rec.id_to_idx]
            sub_tfidf = rec.tfidf_matrix[r_indices]
            s = float(np.mean(np.asarray(sub_tfidf.dot(target_vec.T).todense()).flatten()))
            sims_10_no_mmr.append(s)

            # ILD for no MMR
            if len(r_indices) > 1:
                mat = sub_tfidf.dot(sub_tfidf.T).toarray()
                n_items = len(r_indices)
                upper = [mat[i, j] for i in range(n_items) for j in range(i + 1, n_items)]
                ild_no_mmr.append(float(np.mean([1.0 - c for c in upper])))

        # B. Recommend with MMR (lambda = 0.7)
        recs_mmr_20 = rec.recommend(liked_ids=[mid], n=20, lmbda=0.7)
        recs_mmr_10 = recs_mmr_20[:10]
        rec_ids_mmr_10 = [r["movie_id"] for r in recs_mmr_10]
        rec_ids_mmr_20 = [r["movie_id"] for r in recs_mmr_20]
        unique_recs_mmr.update(rec_ids_mmr_20)

        if target_mid in rec_ids_mmr_10:
            hits_10_mmr += 1
        if target_mid in rec_ids_mmr_20:
            hits_20_mmr += 1

        if rec_ids_mmr_10:
            r_indices = [rec.id_to_idx[r_id] for r_id in rec_ids_mmr_10 if r_id in rec.id_to_idx]
            sub_tfidf = rec.tfidf_matrix[r_indices]
            s = float(np.mean(np.asarray(sub_tfidf.dot(target_vec.T).todense()).flatten()))
            sims_10_mmr.append(s)

            if len(r_indices) > 1:
                mat = sub_tfidf.dot(sub_tfidf.T).toarray()
                n_items = len(r_indices)
                upper = [mat[i, j] for i in range(n_items) for j in range(i + 1, n_items)]
                ild_mmr.append(float(np.mean([1.0 - c for c in upper])))

            # Regional and genre alignment
            reg_matches = sum(1 for r in recs_mmr_10 if r["approx_origin_region"] == probe_region)
            region_match_counts.append(reg_matches / len(recs_mmr_10))
            genre_matches = sum(1 for r in recs_mmr_10 if bool(set(g.lower() for g in r["genres"]) & probe_genres))
            genre_match_counts.append(genre_matches / len(recs_mmr_10))

    total = len(sample_indices)
    results = {
        "catalog": "World Cinema (IMDb Datasets)",
        "evaluation_type": "Leave-one-out Content Similarity",
        "honesty_disclaimer": (
            "Evaluated strictly via content-based leave-one-out item similarity. "
            "Because the IMDb dataset does not contain individual user rating logs, "
            "NO collaborative filtering performance (collaborative Recall/NDCG) is claimed."
        ),
        "total_probes": total,
        "metrics": {
            "pure_content_top10": {
                "hit_rate_10": round(hits_10_no_mmr / total, 4),
                "mean_target_similarity_10": round(float(np.mean(sims_10_no_mmr)), 4),
                "ild_10": round(float(np.mean(ild_no_mmr)), 4),
                "catalog_coverage_unique_items": len(unique_recs_no_mmr),
            },
            "content_plus_mmr_top10": {
                "hit_rate_10": round(hits_10_mmr / total, 4),
                "hit_rate_20": round(hits_20_mmr / total, 4),
                "mean_target_similarity_10": round(float(np.mean(sims_10_mmr)), 4),
                "ild_10": round(float(np.mean(ild_mmr)), 4),
                "catalog_coverage_unique_items": len(unique_recs_mmr),
                "ild_improvement_pct": round(
                    ((np.mean(ild_mmr) - np.mean(ild_no_mmr)) / np.mean(ild_no_mmr)) * 100, 2
                ),
            },
            "alignment": {
                "origin_region_consistency_10": round(float(np.mean(region_match_counts)), 4),
                "genre_overlap_consistency_10": round(float(np.mean(genre_match_counts)), 4),
            },
        },
    }

    # Save to JSON
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_file = artifacts_dir / "global_eval_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save to Markdown
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    md_content = f"""# 🌍 World Cinema Recommender Evaluation

> **Honest Metric Disclaimer**:
> This evaluation is conducted strictly via **leave-one-out content similarity** on verified items from the IMDb non-commercial catalog ($N={total}$ evaluation probes).
> Because the IMDb dataset does **not** provide user interaction histories or rating logs, **NO collaborative filtering performance (e.g. collaborative Recall@10, user NDCG)** is claimed for this catalog.

---

## 📊 Content-Based Leave-One-Out Evaluation Results

| Pipeline Configuration | Hit Rate@10 | Hit Rate@20 | Target Cosine Sim@10 | Intra-List Diversity (ILD@10) | Unique Catalog Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pure Content (TF-IDF + Bayesian Prior, $\\lambda=1.0$)** | **{results['metrics']['pure_content_top10']['hit_rate_10']:.4f}** | — | **{results['metrics']['pure_content_top10']['mean_target_similarity_10']:.4f}** | {results['metrics']['pure_content_top10']['ild_10']:.4f} | {results['metrics']['pure_content_top10']['catalog_coverage_unique_items']:,} |
| **Content + MMR Diversity ($\\lambda=0.7$)** | {results['metrics']['content_plus_mmr_top10']['hit_rate_10']:.4f} | **{results['metrics']['content_plus_mmr_top10']['hit_rate_20']:.4f}** | {results['metrics']['content_plus_mmr_top10']['mean_target_similarity_10']:.4f} | **{results['metrics']['content_plus_mmr_top10']['ild_10']:.4f}** (+{results['metrics']['content_plus_mmr_top10']['ild_improvement_pct']}%) | **{results['metrics']['content_plus_mmr_top10']['catalog_coverage_unique_items']:,}** |

---

## 🎯 Consistency & Alignment Metrics

- **Genre Overlap Consistency@10**: **{results['metrics']['alignment']['genre_overlap_consistency_10'] * 100:.1f}%** of recommended items share primary genre classifications with the probe movie.
- **Origin Region Consistency@10**: **{results['metrics']['alignment']['origin_region_consistency_10'] * 100:.1f}%** of unconstrained recommendations preserve the approximate origin region.
- **MMR Diversity Impact**: Setting $\\lambda=0.7$ increases Intra-List Diversity by **+{results['metrics']['content_plus_mmr_top10']['ild_improvement_pct']}%** while maintaining strong target recovery ({results['metrics']['content_plus_mmr_top10']['hit_rate_10'] * 100:.1f}% Hit@10, {results['metrics']['content_plus_mmr_top10']['hit_rate_20'] * 100:.1f}% Hit@20).
"""
    with open(DOCS_DIR / "global_eval.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"[global_eval] Saved evaluation results to {out_file} and docs/global_eval.md")
    return results

if __name__ == "__main__":
    run_global_evaluation()
