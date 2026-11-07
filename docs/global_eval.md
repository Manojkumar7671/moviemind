# World Cinema Recommender Evaluation

**Evaluation Benchmark**: Honest Content Target Recovery ($N=500$, seed=42) over 33,485 films from IMDb non-commercial datasets with Wikidata (CC0) origin and language metadata.

## Protocol
For each target film $T$:
1. Identify all other films in the catalog sharing at least one genre and at least one original language with $T$.
2. Randomly sample two distinct conditioning films $A$ and $B$.
3. Construct the user taste profile from $A$ and $B$ only.
4. Rank the entire catalog excluding $A$ and $B$.
5. A **Hit** is scored if target film $T$ appears in the top-20 recommendations ($K=20$).

## Results Comparison

| Model / Pipeline | Hit@20 | Hits (500 targets) | Absolute Lift vs Baseline |
| :--- | :---: | :---: | :---: |
| **MovieMind Content Recommender (TF-IDF + Prior + MMR $\lambda=0.7$)** | **5.20%** | **26** | — |
| **Random Ranking Baseline** | 0.00% | 0 | +5.20% |
| **Popularity Ranking Baseline (Most Votes)** | 0.00% | 0 | +5.20% |

- **Lift over Random**: **+5.20%** (5.20% vs 0.00%)
- **Lift over Popularity**: **+5.20%** (5.20% vs 0.00%)

## Honest Limitations
1. **Sanity Check Only**: The target film shares attributes (genre and original language) with the conditioned liked films by construction. This evaluation is an attribute-conditioned retrieval sanity check, not proof of individual user satisfaction or personalization quality.
2. **No Collaborative Filtering**: IMDb non-commercial datasets do not publish user-level viewing or rating histories. No collaborative filtering performance (collaborative Recall@K, NDCG@K) is claimed on this catalog.
3. **Sparse & Multi-Origin Films**: Films with sparse metadata or co-productions with multiple countries are matched across all listed countries.
