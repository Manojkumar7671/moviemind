# 🌍 World Cinema Recommender Evaluation

> **Honest Metric Disclaimer**:
> This evaluation is conducted strictly via **leave-one-out content similarity** on verified items from the IMDb non-commercial catalog ($N=500$ evaluation probes).
> Because the IMDb dataset does **not** provide user interaction histories or rating logs, **NO collaborative filtering performance (e.g. collaborative Recall@10, user NDCG)** is claimed for this catalog.

---

## 📊 Content-Based Leave-One-Out Evaluation Results

| Pipeline Configuration | Hit Rate@10 | Hit Rate@20 | Target Cosine Sim@10 | Intra-List Diversity (ILD@10) | Unique Catalog Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pure Content (TF-IDF + Bayesian Prior, $\lambda=1.0$)** | **0.9080** | — | **0.8374** | 0.2033 | 4,262 |
| **Content + MMR Diversity ($\lambda=0.7$)** | 0.9120 | **0.9580** | 0.8335 | **0.2159** (+6.16%) | **7,627** |

---

## 🎯 Consistency & Alignment Metrics

- **Genre Overlap Consistency@10**: **96.0%** of recommended items share primary genre classifications with the probe movie.
- **Origin Region Consistency@10**: **92.4%** of unconstrained recommendations preserve the approximate origin region.
- **MMR Diversity Impact**: Setting $\lambda=0.7$ increases Intra-List Diversity by **+6.16%** while maintaining strong target recovery (91.2% Hit@10, 95.8% Hit@20).
