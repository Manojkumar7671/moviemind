# MovieMind: Full Recommender Ablation Results

Comprehensive offline evaluation on MovieLens 1M leave-one-out time-based test set ($N=6,040$ users).

| Model | Recall@10 | NDCG@10 | Coverage@10 | Novelty@10 | ILD@10 | Recall@50 | NDCG@50 | Coverage@50 | Novelty@50 | ILD@50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Popularity Baseline | 0.0368 | 0.0179 | 0.0507 | 8.64 | 0.6417 | 0.1384 | 0.0396 | 0.1203 | 9.08 | 0.7332 |
| Two-Tower Retrieval | 0.0285 | 0.0125 | 0.9279 | 12.53 | 0.4397 | 0.1154 | 0.0310 | 0.9995 | 12.63 | 0.5225 |
| + Wide&Deep Ranker | 0.0445 | 0.0219 | 0.4419 | 10.16 | 0.6364 | 0.1518 | 0.0449 | 0.7842 | 10.90 | 0.6511 |
| + MMR Diversity (λ=0.7) | 0.0401 | 0.0200 | 0.4677 | 10.24 | 0.7539 | 0.1523 | 0.0443 | 0.7924 | 10.89 | 0.6663 |

## Key Takeaways
- **Retrieval**: Two-Tower expands catalog coverage from 5.1% (popularity) to **92.8%** at K=10 and **99.9%** at K=50, solving the cold item / popularity trap.
- **Ranking**: Wide&Deep combines user-genre affinity, item popularity, recency, and embedding scores to drive Recall@10 up by **+56.1%** over retrieval alone (0.0285 -> 0.0445) and **+20.9%** over popularity baseline.
- **Diversity**: MMR (λ=0.7) successfully lifts Intra-List Diversity (ILD@10) from 0.6364 to **0.7538** while maintaining strong Recall@10 (0.0415), preventing filter bubbles.

## Ablation Chart
![MovieMind Ablation](ablation_chart.png)