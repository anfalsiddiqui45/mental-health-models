# SBERT + MLP: Semantic Embedding Neural Baseline for Mental-Health Text Classification

## Overview
This experiment evaluates a semantic text representation pipeline using pretrained sentence embeddings combined with a shallow neural classifier. Sentence-level embeddings capture contextual meaning beyond bag-of-words features, while a lightweight Multi-Layer Perceptron performs final classification. The model provides a semantic baseline between classical TF–IDF methods and deeper sequence architectures.

## Objective
- introduce contextual sentence semantics
- reduce lexical sparsity seen in TF–IDF models
- improve minority-class recognition
- benchmark embedding-based neural approaches against linear baselines

## Dataset
- Classes: Normal, Depression, Anxiety, Stress
- Total samples: 7,635
- Task: Supervised multi-class text classification

## Input Representation
Embedding model:
- pretrained Sentence-BERT: all-MiniLM-L6-v2
- fixed-size dense sentence vectors (384 dimensions)

Processing:
- raw sentences directly encoded
- no manual feature engineering required

Result: dense semantic embeddings representing full sentence meaning

## Model Architecture
- Encoder: Sentence-BERT (frozen, inference only)
- Classifier: 2-layer MLP
  - Linear(384 → 256)
  - ReLU
  - Dropout(0.3)
  - Linear(256 → 4)

Loss:
- CrossEntropyLoss with class weights

Class weighting compensates for dataset imbalance, especially Anxiety and Stress.

## Training Configuration
- optimizer: Adam
- learning rate: 1e-3
- epochs: 10
- batch size: 32
- GPU/CPU supported

## Results

### Overall Metrics
| Metric | Value |
|-------|-------|
| Accuracy | 0.89 |
| Macro F1 | 0.83 |
| Weighted F1 | 0.90 |

### Per-Class Performance
| Class | Precision | Recall | F1 | Support |
|------------|-----------|--------|--------|---------|
| Normal | 0.96 | 0.92 | 0.94 | 3269 |
| Depression | 0.95 | 0.89 | 0.92 | 3081 |
| Anxiety | 0.77 | 0.87 | 0.82 | 768 |
| Stress | 0.54 | 0.79 | 0.64 | 517 |

## Findings
- Best overall performance among shallow models
- Significant improvement in Anxiety recall compared to TF–IDF baselines
- Better semantic discrimination between emotional states
- Dense embeddings generalize better to unseen phrasing
- Training remains computationally efficient

## Limitations
- Encoder is frozen (not fine-tuned)
- Slightly heavier inference than linear models
- Stress class precision remains low
- Does not explicitly model sequential token dependencies

## Artifacts
- confusion matrix image
- per-class metrics CSV
- training script in models/sbertlr.py
