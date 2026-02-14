# SBERT + Logistic Regression for Mental-Health Text Classification

## Overview
This experiment implements a **semantic-based linear classification** approach for multi-class mental-health text classification. We use **Sentence-BERT embeddings** to convert each statement into a dense vector representation, which is then classified with a **Logistic Regression** model.

## Objective
- Leverage semantic embeddings to capture context and meaning beyond simple lexical features.
- Evaluate the performance of a lightweight linear classifier using class-weighted logistic regression.
- Compare against classical TF–IDF and neural-based baselines.

## Dataset
- Dataset: Mental Health Text Dataset
- Classes: Normal, Depression, Anxiety, Stress (4 classes)
- Total samples: 7,635
- Task: Supervised multi-class text classification

## Data Preprocessing
- Tokenization and basic cleaning handled internally by Sentence-BERT.
- Sentences are encoded into 384-dimensional embeddings using `all-MiniLM-L6-v2`.
- No additional feature engineering; embeddings capture semantic similarity.

## Model Architecture
- Feature extractor: **:contentReference[oaicite:0]{index=0}**
- Classifier: **:contentReference[oaicite:1]{index=1}**
- Class-weighted to handle dataset imbalance.
- Maximum iterations: 10,000 for convergence.

## Training Details
- Train/Test split as provided in the dataset.
- Class weights calculated automatically from the training set.
- Logistic Regression uses balanced weighting to mitigate class imbalance.

## Evaluation Metrics
- Accuracy
- Macro and Weighted F1
- Per-class Precision, Recall, F1
- Confusion Matrix for error analysis

## Results

| Class       | Precision | Recall | F1-score | Support |
|------------|----------|-------|----------|---------|
| Normal     | 0.95     | 0.91  | 0.93     | 3269    |
| Depression | 0.95     | 0.86  | 0.90     | 3081    |
| Anxiety    | 0.73     | 0.85  | 0.79     | 768     |
| Stress     | 0.47     | 0.78  | 0.59     | 517     |

**Overall Metrics**  

| Metric         | Score    |
|----------------|---------|
| Accuracy       | 0.8732  |
| Macro F1       | 0.8025  |
| Weighted F1    | 0.8813  |
| Macro Precision| 0.7779  |
| Macro Recall   | 0.8494  |

## Analysis and Findings
- Strengths: High semantic awareness enables good performance on Normal and Depression classes.
- Weaknesses: Lower precision for Stress; model sometimes confuses subtle emotional nuances.
- Root causes: Class imbalance and linear classifier limitation in distinguishing minority classes.

## Limitations
- Sentence-BERT embeddings rely on pre-trained semantics; may not fully capture domain-specific language nuances.
- Logistic Regression, even with class weights, has limited capacity to model non-linear relationships.
- Performance depends on embedding quality; more advanced classifiers may improve minority class recognition.

## Artifacts
- Confusion Matrix saved as: `results/sbert_lr_cm.png`
- Classification Report CSV: `results/sbert_lr_metrics.csv`
