# Recommendation System

## Multi-Source Candidate Generation

The system uses a **retrieve-then-rank** architecture to avoid scoring every article with the expensive ranking model.

### 1. Popularity Candidates
- Global top-N by time-decayed click count
- Category-aware top-N for the user's preferred categories
- Source: `popularity_global`, `popularity_category`

### 2. Content-Similarity Candidates
- TF-IDF vectorization of article titles + categories
- Cosine similarity between user history profile and all articles
- Source: `content_similarity`

### 3. Collaborative Candidates
- User-user similarity via Jaccard index on clicked items
- Inverted index for efficient neighbor discovery
- Items consumed by similar users, weighted by similarity
- Source: `collaborative`

### Candidate Pool
- Default: 200 candidates per user
- Deduplicated across sources
- Source tracked for analysis

## Cold-Start Handling

### New Users (no interaction history)
- Fallback to popular content
- Category diversity enforced
- Source: `cold_start_popular`

### New Items (no interaction data)
- Content features still available (title, category)
- Freshness score = 1.0 (maximum)
- Can be scored by content-based features

## Diversity-Aware Reranking (MMR)

After ML ranking, Maximal Marginal Relevance reranking balances relevance and diversity:
```
MMR(d) = λ × Relevance(d) - (1-λ) × max_similarity_to_selected(d)
```
- λ = 0.6 by default (tunable)
- Diversity signal: category overlap with selected items

## Freshness-Aware Ranking

Freshness boost applied before diversity reranking:
```
adjusted_score = score + boost_weight × exp(-age / half_life)
```
- Half-life: 24 hours
- Boost weight: 0.1
