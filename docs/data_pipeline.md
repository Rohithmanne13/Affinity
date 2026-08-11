# Data Pipeline

## Dataset: Microsoft News Dataset (MIND)

MIND is a large-scale news recommendation dataset from Microsoft Research containing:
- **news.tsv**: Article metadata (ID, category, subcategory, title, abstract)
- **behaviors.tsv**: User interaction logs (impression ID, user ID, timestamp, history, impressions)

### Schema

| File | Columns |
|---|---|
| news.tsv | news_id, category, subcategory, title, abstract, url, title_entities, abstract_entities |
| behaviors.tsv | impression_id, user_id, timestamp, history (space-separated news IDs), impressions (news_id-label pairs) |

## Preprocessing Pipeline

### 1. Loading
- PySpark reads TSV files with explicit schemas
- Handles malformed records via `DROPMALFORMED` mode

### 2. Impression Parsing
- Splits `"N12345-1 N67890-0"` into individual (news_id, label) rows
- label=1 → clicked, label=0 → not clicked

### 3. Data Cleaning
- Remove null user_id, news_id, or label
- Remove news IDs not present in the news catalogue
- Optional duplicate removal
- Log all statistics (records removed, users, items, positive/negative ratio)

### 4. Time-Aware Splitting
- **Strategy**: Chronological split using timestamp quantiles
- **Ratios**: 70% train, 15% validation, 15% test
- **Purpose**: Prevents future data leakage into training features

### Leakage Prevention
- User/item features computed from TRAINING data only
- Validation and test sets inherit training-computed features
- No interaction information from the future used in feature construction
