# 🎬 MovieMind: Two-Stage Movie Recommender & Tool-Calling Agent

MovieMind is an end-to-end, two-stage movie recommendation system with a conversational tool-calling agent. Built on the MovieLens 1M dataset, it couples a **Two-Tower Neural Retrieval** model and **Wide & Deep Ranker** with **Maximal Marginal Relevance (MMR)** diversity re-ranking, cold-start onboarding, SQLite taste memory, and an interactive FastAPI chat interface.

---

## ⚡ Quick Start

### 1. Requirements & Setup
- Python 3.11
- Operating System: macOS, Linux, or Windows (CPU-only, no CUDA dependencies)

```bash
# Clone and enter directory
cd moviemind

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install pinned dependencies
pip install -r requirements.txt

# (Optional) Copy environment template for OpenRouter LLM agent
cp .env.example .env
```

### 2. Run the Web Application
Because all trained models, FAISS indices, embeddings, and candidate sets are precomputed in `artifacts/` (< 35 MB total repo size), the app boots instantly with **zero training required**:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
# or: ./run.sh serve
# or: make serve
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser to access the Chat UI and Genre Onboarding.  
Open **[http://localhost:8000/ab](http://localhost:8000/ab)** to view the live A/B Testing Simulation Dashboard.

---

## 📊 Offline Ablation Results

Evaluated on the strict leave-one-out time-based MovieLens 1M test set ($N=6,040$ users; last interaction = test, second-last = validation, rest = train, no data leakage):

| Model Stage | Recall@10 | NDCG@10 | Coverage@10 | Novelty@10 | ILD@10 | Recall@50 | NDCG@50 | Coverage@50 | Novelty@50 | ILD@50 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Popularity Baseline** | 0.0368 | 0.0179 | 0.0507 | 8.64 | 0.6417 | 0.1384 | 0.0396 | 0.1203 | 9.08 | 0.7332 |
| **Two-Tower Retrieval** | 0.0285 | 0.0125 | **0.9279** | **12.53** | 0.4397 | 0.1154 | 0.0310 | **0.9995** | **12.63** | 0.5225 |
| **+ Wide&Deep Ranker** | **0.0445** | **0.0219** | 0.4419 | 10.16 | 0.6364 | 0.1518 | **0.0449** | 0.7842 | 10.90 | 0.6511 |
| **+ MMR Diversity (λ=0.7)** | 0.0401 | 0.0200 | 0.4677 | 10.24 | **0.7539** | **0.1523** | 0.0443 | 0.7924 | 10.89 | 0.6663 |

### Key Findings
1. **Catalog Exploration**: Two-Tower retrieval expands catalog coverage from **5.1%** (popularity) to **92.8%** at K=10 and **99.95%** at K=50.
2. **Ranking Precision**: Wide&Deep ranker delivers a **+20.9% Recall@10 lift** over the popularity baseline and **+56.1%** over retrieval alone.
3. **Diversity Preservation**: MMR ($\lambda=0.7$) elevates Intra-List Diversity (ILD@10) from **0.6364** to **0.7539** with negligible accuracy compromise.

---

## 🔬 Simulated A/B Test (Control vs. Full Pipeline)

Simulated against held-out test data ($N=5,000$ user sessions) with a position-biased click examination model:

- **Control (Popularity)**: 8.48% Item CTR (60.48% Session CTR)
- **Treatment (MovieMind Full Pipeline)**: 9.42% Item CTR (62.88% Session CTR)
- **Relative CTR Lift**: **+11.18%**
- **95% Confidence Interval**: **[+0.0045, +0.0145]**
- **Statistical Significance**: $p = 2.05 \times 10^{-4}$ (Two-proportion z-test, significant at $\alpha = 0.01$)

---

## 🛠 System Design & Architecture

```
User Query / Onboarding
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ FastAPI Serving Layer (app/main.py)                    │
│   ├── /api/chat      -> MovieMindAgent (OpenRouter/FB) │
│   ├── /api/onboarding-> ColdStartRecommender           │
│   └── /ab            -> A/B Testing Dashboard          │
└────────────────────────────────────────────────────────┘
         │
         ├──► Stage 1: Retrieval (Two-Tower InfoNCE + FAISS IndexFlatIP)
         ├──► Stage 2: Heavy Ranking (Wide & Deep with Cross Features)
         ├──► Stage 3: Diversity (MMR Re-ranking, λ=0.7)
         └──► Memory: SQLite Feedback & Taste Profile Store (data/memory.db)
```

### Why PyTorch?
1. **Lightweight & Modular**: PyTorch provides direct tensor-level control without the heavy abstraction layers of TF-Recommenders, allowing the Two-Tower and Wide&Deep architectures to be cleanly defined in under 100 lines each.
2. **CPU-First Vectorization**: PyTorch tensors natively integrate with NumPy and FAISS, enabling rapid vector operations, batch dot-products, and training without GPU/CUDA overhead.
3. **Compact Weights**: The entire trained model footprint is less than 3 MB, making containerization and cold booting effortless.

---

## 🧪 Testing

Execute the comprehensive unit and integration test suite:

```bash
pytest -v
# or: ./run.sh test
```

Verified test coverage includes:
- Strict time-based split verification (zero future interaction leakage)
- Exact mathematical calculation of Recall@K, NDCG@K, Coverage, Novelty, and ILD
- MMR Diversity monotonicity ($\text{ILD}_{\text{MMR}} \ge \text{ILD}_{\text{no-MMR}}$)
- Cold-start recommendation shape and field contracts
- FastAPI endpoint smoke tests

---

## 🐳 Docker & Hugging Face Spaces Deployment

Build and run locally with Docker:
```bash
docker build -t moviemind .
docker run -p 8000:8000 moviemind
```

To deploy to **Hugging Face Spaces**, create a Space using the Docker SDK and follow the instructions in [`docs/hf_spaces.md`](docs/hf_spaces.md).
