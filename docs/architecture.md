# MovieMind Architecture

MovieMind implements a production-grade, two-stage recommendation system augmented with an LLM tool-calling conversational agent.

## System Topology & Flow

```mermaid
flowchart TD
    subgraph Data Layer
        ML1M["MovieLens 1M Raw"] --> Split["Strict Time-Based Split\n(Train / Val / Test)"]
        Split --> Parquet["Parquet Data Store\n(train, test, movies, users)"]
    end

    subgraph Offline Stage 1: Retrieval
        Parquet --> TowerTrain["Two-Tower Neural Network\n(User Tower & Item Tower)"]
        TowerTrain --> ItemEmb["Catalog Item Embeddings (64-d)"]
        ItemEmb --> Faiss["FAISS IndexFlatIP Index"]
        TowerTrain --> UserEmb["User Representation Vector"]
        UserEmb & Faiss --> Top200["Top-200 Candidate Generation"]
    end

    subgraph Offline Stage 2: Ranking
        Top200 --> WideDeep["Wide & Deep Ranker\n(Genre Affinity, Popularity, Recency, Tower Score)"]
        WideDeep --> RankedCandidates["Top-50 Scored Candidates"]
    end

    subgraph Offline Stage 3: Diversity
        RankedCandidates --> MMR["Maximal Marginal Relevance (MMR)\n(λ=0.7 Re-ranking)"]
        MMR --> Artifacts["Saved Artifacts\n(artifacts/)"]
    end

    subgraph Cold Start Flow
        NewUser["New User / Onboarding"] --> GenreSelect["Select 3-5 Favorite Genres"]
        GenreSelect --> ColdModel["ColdStartRecommender\n(Mean Item Embeddings + Genre Popularity)"]
    end

    subgraph Online Serving & Agent Layer
        UserUI["Web UI / Chat Client"] <--> FastAPIServer["FastAPI Backend (app.main)"]
        FastAPIServer <--> Agent["MovieMindAgent (OpenRouter LLM / Fallback)"]
        Agent <--> Tools["Agent Tools:\n- search_movies\n- recommend\n- explain\n- filter_movies\n- get_taste_profile\n- record_feedback"]
        Tools <--> SQLite[("SQLite User Memory\n(data/memory.db)")]
        Tools <--> Artifacts
        Tools <--> ColdModel
        FastAPIServer --> ABDashboard["/ab Simulation Dashboard"]
    end
```

## Architecture Stages

1. **Candidate Retrieval (Two-Tower Model)**:
   - **User Tower**: Consumes user ID embedding, demographics (gender, binned age, occupation), and historical mean rating strictly computed on train split.
   - **Item Tower**: Consumes movie ID embedding, 18-dim multi-hot genre projection, and normalized release year.
   - **Loss**: InfoNCE with in-batch negatives.
   - **Vector Index**: FAISS `IndexFlatIP` indexing L2-normalized 64-dimensional vectors. Sub-millisecond candidate retrieval.

2. **Heavy Ranking (Wide & Deep Model)**:
   - **Wide Component**: Linear model capturing explicit cross-features: user-genre historical affinity, item popularity, recency, and retrieval tower score.
   - **Deep Component**: User & item entity embeddings combined with dense cross-feature activations passed through a deep multi-layer perceptron.
   - Re-ranks candidate pools from 200 down to high-precision ranked lists.

3. **Diversity Re-ranking (MMR)**:
   - Balances relevance $S_{\text{norm}}(i)$ with item-to-item genre similarity:
     $$\text{MMR}(i) = \lambda \cdot S_{\text{norm}}(i) - (1 - \lambda) \cdot \max_{j \in R} \text{sim}(i, j)$$
   - Tunable $\lambda \in [0, 1]$. At $\lambda = 0.7$, elevates Intra-List Diversity (ILD) while preserving top-tier Recall and NDCG.

4. **Conversational Tool-Calling Agent**:
   - Uses OpenAI client pointed at OpenRouter (e.g. `meta-llama/llama-3.1-8b-instruct:free`).
   - Equipped with 6 typed tools to query catalog, generate recommendations, filter by release date/genre, fetch taste summaries, and record user likes/dislikes.
   - Factual grounding: explanations cite exact shared genres and similarity to liked movies; zero hallucination.
   - Autonomous fallback router ensures offline and API-independent operation.
