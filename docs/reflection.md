# System Design & Implementation Reflection

## 1. Offline vs. Online Alignment
- **Offline Evaluation**: Evaluated using leave-one-out time-based split where each user's latest interaction is withheld for testing, and the second-latest for validation. Leakage was eliminated by computing all user history features, movie popularity, and user genre distributions strictly on training interactions occurring before the validation and test timestamps.
- **Online Simulation (A/B Testing)**: A synthetic click model incorporating position examination bias ($1 / \log_2(r + 1)$) and user-item attraction (ground truth target match + genre affinity) demonstrated a **+11.18% relative CTR lift** ($p = 2.05 \times 10^{-4}$) for the two-stage pipeline over global popularity.

## 2. Multi-Objective Tradeoffs: Accuracy vs. Diversity
- A pure relevance ranker tends to concentrate recommendations in a user's dominant genre or top popular movies (filter bubble effect).
- Implementing Maximal Marginal Relevance (MMR) across varying values of $\lambda \in [0.0, 1.0]$ exposed the Pareto frontier:
  - $\lambda = 1.0$ yields maximum Recall@10 (0.0445) but lowest ILD@10 (0.6364).
  - $\lambda = 0.7$ strikes the sweet spot, elevating ILD@10 to **0.7539** while maintaining strong Recall@10 (**0.0401**).

## 3. Cold-Start Strategy
- For brand-new users with zero historical interactions, user ID embeddings cannot be utilized.
- MovieMind solves cold start via explicit genre onboarding:
  1. The user selects 3 to 5 preferred genres.
  2. A pseudo-user embedding is generated from the normalized mean embedding of top catalog items in those genres.
  3. A hybrid scoring function merges vector embedding similarity with intra-genre popularity to present high-confidence recommendations immediately.

## 4. Latency & Resource Efficiency
- **Precomputation**: By exporting 64-dimensional item vectors and prebuilding a FAISS `IndexFlatIP` index (~971 KB), candidate generation executes in sub-millisecond time on CPU.
- **Repository Footprint**: Entire repository with all trained model weights, FAISS index, parquet splits, and candidates totals ~33 MB, well under the 100 MB budget.
- **Inference Without Training**: The FastAPI service and chat agent start instantly without executing any training steps.

## 5. Potential Extensions
- **Real-Time Sequential Retrieval**: Integrating SASRec or a Transformer-based sequence encoder over the user's last $N$ clicked items stored in `memory.db`.
- **Bandit Exploration**: Applying LinUCB or Thompson Sampling on top of candidate pools to explore user openness to novel genres.
