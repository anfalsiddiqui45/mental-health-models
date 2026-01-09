# Mental Health Text Classification Under Model Constraints

## Problem Definition
This project aims to classify user generated textual data into Mental Health Condition Indicators (stress ,depression and anxiety).
Given the sensitive nature of the domain, the goal is not only to achieve high predictive performance, but also to analyze model behavior under practical constraints.
We are implementing different machine learning and deep learning architectures to observe different parameters in our case.
Model evaluation considers multiple parameters including **Accuracy**, **Precision**, **Recall**, **F1-score**, **Performance on unseen data** and **Confusion matrices**, along with qualitative analysis of failure cases and generalization behavior.

Multiple machine learning and deep learning architectures are evaluated to study their performance, generalization ability, and failure modes in this context.



## Problem Complexity and Challenges
- Overlapping linguistic cues
- Short, informal text
- High semantic ambiguity
- Ethical risk of false positives

## Constraints
- Limited labeled data
- Moderate compute
- Inference latency matters
- Safety > raw accuracy

## Baseline
We start with Logistic Regression + TF-IDF to establish a transparent baseline.

## Models Considered (Before Experiments)
- Logistic Regression (interpretability)
- RNN (sequence awareness)
- Bi-LSTM (long-range context)
- Transformer (semantic richness)

## What We Expect to Fail
- Short texts with mixed emotions
- Sarcasm
- Indirect expressions
- Overfitting on small datasets
