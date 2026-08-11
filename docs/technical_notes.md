# Technical Notes

## Key Design Decisions

### Why XGBoost `rank:pairwise` over classification?
- Learning-to-rank objectives directly optimize for ranking quality
- Pairwise loss considers relative ordering within impression groups
- More appropriate for recommendation ranking than pointwise BCE

### Why K-Means for user segmentation?
- Interpretable cluster assignments
- Efficient for the scale of MIND-small
- Cluster ID as a feature adds segment-level signals to the ranker

### Why TF-IDF for content similarity?
- Lightweight and effective for news titles
- No need for pre-trained embeddings for this scope
- Fast indexing and retrieval

### Why MMR for diversity?
- Principled approach with tunable relevance-diversity trade-off
- Category-based diversity is natural for news recommendation
- Avoids ad-hoc heuristics

### Leakage Prevention Strategy
1. Time-aware chronological split (no random shuffle)
2. User/item features computed only from training interactions
3. Context features use only the interaction's own timestamp
4. No global statistics computed on test data

### Scalability Design
- Spark pipeline designed for `local[*]` but scales to cluster mode
- Configurable partitions and shuffle partitions
- Feature pipeline can handle MIND-large with increased memory
- API designed for single-request latency optimization
