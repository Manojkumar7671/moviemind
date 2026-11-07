# World Cinema Recommender Evaluation

**Evaluation Benchmark**: Fair Content Target Recovery ($N=500$, seed=42) over 33,485 films from IMDb non-commercial datasets with Wikidata (CC0) origin and language metadata.

## Protocol
For each target film $T$:
1. Identify all other films in the catalog sharing at least one genre and at least one original language with $T$.
2. Randomly sample two distinct conditioning films $A$ and $B$.
3. Construct the user taste profile from $A$ and $B$ only.
4. Define the candidate pool: all films sharing $\ge 1$ genre and $\ge 1$ language with $A$ or $B$, excluding $A$ and $B$.
5. A **Hit** is scored if target film $T$ appears in the top-20 recommendations ($K=20$).

## Results Comparison

| Model / Pipeline | Candidate Pool | Hit@20 | Hits (500 targets) | 95% Wilson CI |
| :--- | :--- | :---: | :---: | :---: |
| **MovieMind Content Recommender (TF-IDF + Prior + MMR $\lambda=0.7$)** | Full Catalog (profile-guided) | **5.20%** | **26** | **[3.57%, 7.51%]** |
| **In-Pool Random Baseline** | Same Candidate Pool (shared genre & lang) | 2.40% | 12 | [1.38%, 4.15%] |
| **In-Pool Most-Votes Baseline** | Same Candidate Pool (shared genre & lang) | 1.60% | 8 | [0.81%, 3.13%] |
| **Global Most-Votes Baseline (Unconditioned)** | Full Catalog (unconditioned) | 0.00% | 0 | [0.00%, 0.76%] |
| **Global Random Baseline (Unconditioned)** | Full Catalog (unconditioned) | 0.00% | 0 | [0.00%, 0.76%] |

### Fair Comparison Notes
- **Candidate Pool Fairness**: Because target films share attributes (genre and language) with the conditioning films by construction, comparing against unconditioned global baselines (which sample across all 33,485 films without knowing the conditioned attributes) produces an uninformative 0.00% hit rate. Claims of lift over unconditioned baselines have been removed.
- **In-Pool Performance**: Within the exact candidate pool of films matching the conditioning attributes (mean pool size $\sim 10,360$ films), the content recommender achieves **5.20% Hit@20** (95% CI: [3.57%, 7.51%]), significantly exceeding the in-pool most-votes baseline (**1.60%**, 95% CI: [0.81%, 3.13%]) and the in-pool random baseline (**2.40%**, 95% CI: [1.38%, 4.15%]).

## Honest Limitations
1. **Attribute-Conditioned Retrieval Benchmark**: The target film shares attributes with the liked films by construction. This is a content-similarity retrieval sanity check, not evidence of real-world user satisfaction.
2. **No Collaborative Filtering**: IMDb non-commercial datasets do not publish user-level viewing or rating histories. No collaborative filtering metrics (Recall@K, NDCG@K) are claimed.
3. **Sparse & Multi-Origin Films**: Co-productions with multiple countries or languages are matched across all listed attributes.
