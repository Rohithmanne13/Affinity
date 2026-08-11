# Evaluation

## Metrics

### Precision@K
Fraction of top-K recommended items that are relevant.

### Recall@K
Fraction of all relevant items that appear in the top-K.

### NDCG@K (Normalized Discounted Cumulative Gain)
Measures ranking quality — rewards relevant items appearing higher.

### Evaluation at K = 5 and K = 10.

## Methodology

- **Same test set** for all models
- **Group-based evaluation**: metrics computed per impression, then averaged
- **Time-aware split**: test data is chronologically after training data
- **No future leakage**: features use only historical information

## Diversity Metrics

- **Category Coverage**: fraction of all categories represented
- **Intra-List Diversity (ILD)**: fraction of item pairs with different categories
