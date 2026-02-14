# TF-IDF + Linear SVM for Mental-Health Text Classification

## Overview
This experiment implements a **classic feature-based linear classification** approach for multi-class mental-health text classification. We use **TF–IDF features** extracted from statements and classify them using a **Linear Support Vector Machine (SVM)**.

## Objective
- Leverage lexical n-gram features for robust text representation.
- Evaluate performance of a linear SVM with class weighting to handle imbalanced data.
- Compare classical TF–IDF + SVM performance with embedding-based and neural baselines.

## Dataset
- Dataset: Mental Health Text Dataset
- Classes: Normal, Depression, Anxiety, Stress (4 classes)
- Total samples: 7,635
- Task: Supervised multi-class text classification

## Data Preprocessing
- Statements are converted to lowercase.
- Stop words removed.
- Features extracted using **TF–IDF**, considering unigrams and bigrams.
- Maximum 50,000 features retained.

## Model Architecture
- Feature extractor: **TF–IDF vectorization**
- Classifier: **Linear Support Vector Machine (SVM)**
- Class weights: Balanced to account for class imbalance.
- Maximum iterations: 10,000 to ensure convergence.

## Training Details
- Train/Test split as provided.
- LinearSVC trained on TF–IDF feature matrix.
- No additional feature engineering required.

## Evaluation Metrics
- Accuracy
- Macro and Weighted F1
- Per-class Precision, Recall, F1
- Confusion Matrix for detailed error analysis

## Results

| Class       | Precision | Recall | F1-score | Support |
|------------|----------|-------|----------|---------|
| Normal     | 0.92     | 0.94  | 0.93     | 3269    |
| Depression | 0.91     | 0.91  | 0.91     | 3081    |
| Anxiety    | 0.83     | 0.82  | 0.83     | 768     |
| Stress     | 0.72     | 0.62  | 0.67     | 517     |

**Overall Metrics**  

| Metric         | Score    |
|----------------|---------|
| Accuracy       | 0.90    |
| Macro F1       | 0.83    |
| Weighted F1    | 0.90    |
| Macro Precision| 0.85    |
| Macro Recall   | 0.82    |

## Analysis and Findings
- Strengths: Excellent performance on Normal and Depression classes.
- Weaknesses: Moderate recall for Stress; class imbalance affects minority classes.
- Insights: TF–IDF captures lexical patterns well, but linear SVM may miss subtle semantic nuances captured by embeddings.

## Limitations
- Limited semantic understanding compared to embedding-based models.
- Class imbalance affects performance on minority classes.
- Linear model may underperform for complex patterns requiring non-linear decision boundaries.

## Artifacts
- Confusion Matrix saved as: `results/tfidf_svm_cm.png`
- Classification Report CSV: `results/tfidf_svm_metrics.csv`
