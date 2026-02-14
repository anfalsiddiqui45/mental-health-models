# CNN + LSTM with Data Augmentation: Deep Learning Evaluation for Mental-Health Text Classification

## Study Summary
This document reports the evaluation of a hybrid Convolutional Neural Network and Long Short-Term Memory architecture trained with text data augmentation for four-class mental-health text classification. The model captures both local n-gram patterns through convolution and long-range contextual dependencies through recurrent memory units, providing a semantic alternative to classical TF–IDF baselines.

## Objective
Assess whether deep sequential modeling combined with augmented training samples improves minority-class generalization and contextual understanding compared to sparse lexical representations.

## Dataset
Dataset: Mental Health Text Dataset  
Classes: Normal, Depression, Anxiety, Stress  
Total samples: 7,635  
Task: Multi-class supervised text classification

## Input Processing and Feature Handling
Text is lower-cased and cleaned prior to tokenization. Sentences are converted to integer sequences using a vocabulary tokenizer and padded to fixed length. Word embeddings are learned during training. Data augmentation is applied to synthetically expand training samples and reduce class imbalance effects.

## Model Architecture
Embedding Layer → 1D Convolution → Max Pooling → LSTM → Fully Connected Layer → Softmax  
CNN extracts local phrase-level patterns while LSTM models temporal and contextual dependencies across the sequence. The hybrid design enables both syntactic and semantic feature learning.

## Training Configuration
Loss: Cross-Entropy  
Optimizer: Adam  
Evaluation: Accuracy, Precision, Recall, F1-Score, Confusion Matrix  
Regularization: implicit through augmentation and dropout (if enabled in script)

## Results

### Overall Metrics

| Metric | Value |
|-------|--------|
| Accuracy | 0.8800 |
| Macro F1 | 0.7979 |
| Weighted F1 | 0.8800 |

### Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|--------|-----------|-----------|-----------|-----------|
| Normal | 0.95 | 0.91 | 0.93 | 3269 |
| Depression | 0.88 | 0.93 | 0.91 | 3081 |
| Anxiety | 0.84 | 0.75 | 0.79 | 768 |
| Stress | 0.55 | 0.60 | 0.57 | 517 |

Confusion matrix saved as: confusion_matrix.png

## Findings
The model achieves strong performance on Normal and Depression classes with balanced precision and recall, indicating effective contextual learning. Anxiety performance is moderate, suggesting partial improvement over lexical baselines but still limited by semantic overlap. Stress remains the weakest class, reflecting insufficient discriminative signals and persistent imbalance despite augmentation. Overall accuracy matches classical models while providing better contextual representation.

## Limitations
Training cost and inference time are higher than TF–IDF baselines. Performance remains sensitive to class imbalance. Deep models require larger data volumes for consistent gains and may overfit without careful regularization.

## Artifacts
Training script: cnn+lstm(dataarg).py  
Outputs: confusion_matrix.png, metrics report, saved model weights  
Location: evaluation/cnn_lstm_aug/
