# LR + TF–IDF Baseline
Classical sparse text classification baseline using TF–IDF features with a Logistic Regression classifier. This model provides a fast, interpretable reference point for evaluating embedding-based and deep neural architectures.

## Dataset
Mental-health text dataset with 4 labels: Normal, Depression, Anxiety, Stress. Total samples: 7,635. Train/test splits loaded from processed/train.csv and processed/test.csv.

## Input Representation
Raw statements → lowercase cleaning → TF–IDF vectorization (unigram + bigram, max_features=20k, english stopwords) → sparse matrix.

## Model Configuration
Logistic Regression, max_iter=1000, class_weight=balanced, linear decision boundary.

## Evaluation Metrics
Accuracy, per-class precision/recall/F1, macro F1, weighted F1, confusion matrix.
## Results

### Overall Metrics

| Metric | Value |
|--------|---------|
| Accuracy | 0.8821 |
| Macro F1 | 0.8216 |
| Weighted F1 | 0.8849 |

### Per-Class Performance

| Class | Precision | Recall | F1-score | Support |
|---------|-----------|-----------|-----------|-----------|
| Normal | 0.9208 | 0.9217 | 0.9213 | 3269 |
| Depression | 0.9367 | 0.8741 | 0.9043 | 3081 |
| Anxiety | 0.7799 | 0.8398 | 0.8088 | 768 |
| Stress | 0.5809 | 0.7427 | 0.6520 | 517 |

## Limitations
No contextual semantics, sparse high-dimensional vectors, sensitive to wording variation, linear classifier capacity only.
High performance on majority classes (Normal, Depression). Moderate results for Anxiety. Lowest precision on Stress. Class imbalance and lexical-only features reduce discrimination between emotionally similar categories.

