# Hierarchical FastText + Lexicon + CNN–LSTM: Multi-Stage Mental-Health Text Classification

## Overview
This document reports a hierarchical multi-stage classification architecture for mental-health text detection. Instead of solving the 4-class problem in a single step, the task is decomposed into two sequential stages to reduce class confusion and improve separation between Normal and Distress categories. The approach combines lexical signals, shallow embeddings, classical machine learning, and deep neural networks within a unified pipeline.

## Objective
Reduce misclassification between semantically overlapping classes and improve minority-class performance by:
- isolating Normal vs Distress at Stage 1
- specializing emotion classification at Stage 2
- combining symbolic (lexicon), distributional (FastText), and neural (CNN–LSTM) features

## Dataset
- Classes: Normal, Depression, Anxiety, Stress
- Total samples: 7,635
- Stage 1 samples: 7,635
- Stage 2 samples (Distress only): 4,366
- Task: Supervised multi-class text classification

## Input Representation
Text preprocessing:
- lowercase conversion
- URL removal
- punctuation removal
- tokenization using NLTK

Feature construction:
- FastText sentence embedding (100-dimensional average word vectors)
- Lexicon features:
  - depression keyword count
  - anxiety keyword count
  - stress keyword count
  - sentence length
- Final feature vector = embedding ⊕ lexicon features

## Architecture

### Stage 1 — Normal vs Distress
- Model: Logistic Regression
- Class weights: balanced
- Purpose: remove Normal early to reduce surface-level lexical confusion with Depression

### Stage 2 — Distress subclassification
Two approaches were evaluated:

1. Classical SVM (RBF kernel)
2. Hybrid CNN–LSTM neural network

Final Stage 2 model:
- CNN layers for local feature extraction
- BiLSTM for sequential dependency modeling
- Feature fusion (CNN + LSTM outputs)
- Fully connected classifier
- Class-weighted loss to handle imbalance

## Rationale for Hierarchical Design
Single-stage classifiers often confuse Normal with Depression due to shared vocabulary. By separating Normal first, Stage 2 focuses only on emotionally similar classes, simplifying decision boundaries and improving recall for minority categories. This divide-and-conquer strategy reduces complexity and yields better specialization.

## Results

### Stage 1 — Normal vs Distress
| Metric | Value |
|-------|-------|
| Accuracy | 0.91 |
| Macro F1 | 0.91 |

| Class | Precision | Recall | F1 | Support |
|--------|-----------|--------|--------|---------|
| Normal | 0.89 | 0.91 | 0.90 | 3269 |
| Distress | 0.93 | 0.91 | 0.92 | 4366 |

---

### Stage 2 — SVM Baseline (Distress only)
| Metric | Value |
|-------|-------|
| Accuracy | 0.53 |
| Macro F1 | 0.4691 |

| Class | Precision | Recall | F1 | Support |
|------------|-----------|--------|--------|---------|
| Depression | 0.89 | 0.53 | 0.66 | 3081 |
| Anxiety | 0.42 | 0.40 | 0.41 | 768 |
| Stress | 0.21 | 0.74 | 0.33 | 517 |

---

### Stage 2 — Hybrid CNN–LSTM (Final)
| Metric | Value |
|-------|-------|
| Accuracy | 0.88 |
| Macro F1 | 0.8035 |
| Weighted F1 | 0.88 |

| Class | Precision | Recall | F1 | Support |
|------------|-----------|--------|--------|---------|
| Depression | 0.96 | 0.91 | 0.94 | 3081 |
| Anxiety | 0.79 | 0.81 | 0.80 | 768 |
| Stress | 0.61 | 0.76 | 0.67 | 517 |

## Findings
- Stage 1 reliably separates Normal with high precision and recall
- SVM struggles with minority subclasses and produces unstable boundaries
- Hybrid CNN–LSTM significantly improves subclass discrimination
- FastText embeddings + lexicon counts provide complementary signals
- Stress remains the most difficult class due to limited samples and semantic overlap

## Limitations
- Hierarchical pipeline increases training complexity
- Error propagation from Stage 1 cannot be corrected in Stage 2
- FastText embeddings are shallow and context-independent
- CNN–LSTM uses sentence-level representation only (no token sequences)

## Artifacts
- confusion_matrix_stage1.png
- confusion_matrix_stage2_svm.png
- confusion_matrix_stage2_cnn_lstm.png
- training scripts in models/hierarchical_classification.py
