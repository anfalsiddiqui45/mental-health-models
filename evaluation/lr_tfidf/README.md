# Logistic Regression + TF–IDF: Classical Baseline for Mental-Health Text Classification

## Overview
This document reports a classical linear baseline using TF–IDF feature representations with Logistic Regression for four-class mental-health text classification. The model serves as a lightweight, interpretable reference for comparing semantic embedding and deep-learning architectures in later experiments.

## Objective
Establish a fast and reproducible baseline to:
- quantify achievable performance with lexical features only
- evaluate class imbalance effects
- provide a benchmark for SVM, SBERT, MLP, CNN–LSTM, and hierarchical models

## Dataset
- Classes: Normal, Depression, Anxiety, Stress
- Total samples: 7,635
- Task: Supervised multi-class text classification

## Input Representation
Text processing:
- lowercase conversion
- stopword removal
- tokenization at word level

Feature extraction:
- TF–IDF weighting
- unigrams + bigrams
- max_features = 20,000
- sparse high-dimensional vectors

Result: sparse document–term matrix used directly as classifier input

## Model Architecture
- Feature extractor: TF–IDF
- Classifier: Logistic Regression (multinomial)
- Optimization: default solver
- Class handling: class_weight = balanced

Class weighting increases the loss contribution of minority classes (Anxiety, Stress) to mitigate imbalance.

## Training Configuration
- Train/test split provided by dataset
- max_iter = 1000
- parallel processing enabled
- no pretrained embeddings used

## Results

### Overall Metrics
| Metric | Value |
|-------|-------|
| Accuracy | 0.88 |
| Macro F1 | 0.82 |
| Weighted F1 | 0.88 |

### Per-Class Performance
| Class | Precision | Recall | F1 | Support |
|------------|-----------|--------|--------|---------|
| Normal | 0.92 | 0.92 | 0.92 | 3269 |
| Depression | 0.94 | 0.87 | 0.90 | 3081 |
| Anxiety | 0.78 | 0.84 | 0.81 | 768 |
| Stress | 0.58 | 0.74 | 0.65 | 517 |

## Findings
- Strong performance on Normal and Depression
- Acceptable recall for Anxiety
- Lower precision for Stress due to vocabulary overlap with other distress classes
- Stable and computationally efficient training
- Suitable as a baseline benchmark

## Limitations
- No contextual semantics
- Bag-of-words assumption ignores word order
- Sparse features limit generalization to unseen phrasing
- Minority classes remain harder despite class weighting

## Artifacts
- results/lr_tfidf_metrics.csv
- results/lr_tfidf_cm.png
- training script in models/baseline_lr_tfdf.ipynb
