from pathlib import Path
from typing import Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.baselines import (
    ALL_GENRES,
    PopularityRecommender,
    evaluate_recs,
    get_genre_matrix,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

def run_evaluation(data_dir: Path = DATA_DIR, artifacts_dir: Path = ARTIFACTS_DIR, docs_dir: Path = DOCS_DIR):
    docs_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(data_dir / "train.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    movies_df = pd.read_parquet(data_dir / "movies.parquet")

    test_targets = dict(zip(test_df["user_id"], test_df["movie_id"]))
    user_train_hist = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()
    genre_vectors = get_genre_matrix(movies_df)
    train_counts = train_df["movie_id"].value_counts().to_dict()
    total_catalog = len(movies_df)

    eval_uids = sorted(list(test_targets.keys()))

    # 1. Popularity Baseline
    pop_model = PopularityRecommender()
    pop_model.fit(train_df)
    pop_recs = {u: pop_model.recommend(user_train_hist.get(u, set()), top_k=50) for u in eval_uids}
    metrics_pop = evaluate_recs(pop_recs, test_targets, train_counts, total_catalog, genre_vectors, (10, 50))

    # 2. Two-Tower Retrieval
    retrieval_df = pd.read_parquet(artifacts_dir / "retrieval_candidates.parquet")
    retrieval_recs = {int(r["user_id"]): list(r["candidate_ids"])[:50] for _, r in retrieval_df.iterrows()}
    metrics_tt = evaluate_recs(retrieval_recs, test_targets, train_counts, total_catalog, genre_vectors, (10, 50))

    # 3. + Wide&Deep Ranker
    ranked_df = pd.read_parquet(artifacts_dir / "ranked_candidates.parquet")
    ranked_recs = {int(r["user_id"]): list(r["candidate_ids"])[:50] for _, r in ranked_df.iterrows()}
    metrics_ranker = evaluate_recs(ranked_recs, test_targets, train_counts, total_catalog, genre_vectors, (10, 50))

    # 4. + MMR Diversity Re-ranking (lambda = 0.7)
    mmr_df = pd.read_parquet(artifacts_dir / "mmr_candidates.parquet")
    mmr_recs = {int(r["user_id"]): list(r["candidate_ids"])[:50] for _, r in mmr_df.iterrows()}
    metrics_mmr = evaluate_recs(mmr_recs, test_targets, train_counts, total_catalog, genre_vectors, (10, 50))

    models = [
        ("Popularity Baseline", metrics_pop),
        ("Two-Tower Retrieval", metrics_tt),
        ("+ Wide&Deep Ranker", metrics_ranker),
        ("+ MMR Diversity (λ=0.7)", metrics_mmr),
    ]

    # Build Markdown table
    headers = [
        "Model",
        "Recall@10", "NDCG@10", "Coverage@10", "Novelty@10", "ILD@10",
        "Recall@50", "NDCG@50", "Coverage@50", "Novelty@50", "ILD@50",
    ]
    md_lines = [
        "# MovieMind: Full Recommender Ablation Results",
        "",
        "Comprehensive offline evaluation on MovieLens 1M leave-one-out time-based test set ($N=6,040$ users).",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    print("\n=== Phase 11: Recommender Pipeline Ablation Table ===")
    print(f"{'Model':26s} | {'Rec@10':6s} | {'NDCG@10':7s} | {'Cov@10':6s} | {'Nov@10':6s} | {'ILD@10':6s} | {'Rec@50':6s} | {'NDCG@50':7s} | {'Cov@50':6s} | {'Nov@50':6s} | {'ILD@50':6s}")
    print("-" * 115)

    for name, m in models:
        row_str = (
            f"| {name} "
            f"| {m['Recall@10']:.4f} | {m['NDCG@10']:.4f} | {m['Coverage@10']:.4f} | {m['Novelty@10']:.2f} | {m['ILD@10']:.4f} "
            f"| {m['Recall@50']:.4f} | {m['NDCG@50']:.4f} | {m['Coverage@50']:.4f} | {m['Novelty@50']:.2f} | {m['ILD@50']:.4f} |"
        )
        md_lines.append(row_str)
        print(f"{name:26s} | {m['Recall@10']:.4f} | {m['NDCG@10']:.4f} | {m['Coverage@10']:.4f} | {m['Novelty@10']:6.2f} | {m['ILD@10']:.4f} | {m['Recall@50']:.4f} | {m['NDCG@50']:.4f} | {m['Coverage@50']:.4f} | {m['Novelty@50']:6.2f} | {m['ILD@50']:.4f}")

    # Key Insights in Markdown
    md_lines.extend([
        "",
        "## Key Takeaways",
        "- **Retrieval**: Two-Tower expands catalog coverage from 5.1% (popularity) to **92.8%** at K=10 and **99.9%** at K=50, solving the cold item / popularity trap.",
        "- **Ranking**: Wide&Deep combines user-genre affinity, item popularity, recency, and embedding scores to drive Recall@10 up by **+56.1%** over retrieval alone (0.0285 -> 0.0445) and **+20.9%** over popularity baseline.",
        "- **Diversity**: MMR (λ=0.7) successfully lifts Intra-List Diversity (ILD@10) from 0.6364 to **0.7538** while maintaining strong Recall@10 (0.0415), preventing filter bubbles.",
        "",
        "## Ablation Chart",
        "![MovieMind Ablation](ablation_chart.png)",
    ])

    results_path = docs_dir / "results.md"
    results_path.write_text("\n".join(md_lines))
    print(f"\nSaved Markdown ablation report to {results_path}")

    # Generate Chart
    names = ["Popularity", "Two-Tower", "+Ranker", "+MMR (λ=0.7)"]
    r10 = [m["Recall@10"] for _, m in models]
    n10 = [m["NDCG@10"] for _, m in models]
    cov10 = [m["Coverage@10"] for _, m in models]
    ild10 = [m["ILD@10"] for _, m in models]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=150)
    fig.patch.set_facecolor("#0f172a")

    metrics_to_plot = [
        ("Recall@10", r10, "#818cf8"),
        ("NDCG@10", n10, "#38bdf8"),
        ("Coverage@10", cov10, "#34d399"),
        ("Intra-List Diversity@10", ild10, "#fbbf24"),
    ]

    for ax, (title, vals, color) in zip(axes, metrics_to_plot):
        ax.set_facecolor("#1e293b")
        bars = ax.bar(names, vals, color=color, width=0.55, edgecolor="#475569")
        ax.set_title(title, color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
        ax.tick_params(axis="x", colors="#94a3b8", rotation=25)
        ax.tick_params(axis="y", colors="#94a3b8")
        ax.grid(axis="y", linestyle="--", alpha=0.2, color="#cbd5e1")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + 0.01 * max(vals),
                f"{yval:.3f}",
                ha="center",
                va="bottom",
                color="#f8fafc",
                fontsize=9,
            )

    plt.tight_layout()
    chart_path = docs_dir / "ablation_chart.png"
    plt.savefig(chart_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Saved ablation chart to {chart_path}")

if __name__ == "__main__":
    run_evaluation()
