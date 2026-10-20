```markdown
# Project 5: K-Nearest Neighbors (KNN) Classification

A supervised machine learning model built to classify Iris flower species based on physical measurements.

## 🧠 Core Concepts
* **Data Splitting:** 80% Training / 20% Testing to prevent overfitting.
* **Feature Scaling:** `StandardScaler` is applied so all features contribute equally to the distance math (critical for KNN).
* **KNN Algorithm:** Classifies data points by finding the 5 nearest neighbors in the feature space.
* **Evaluation:** Uses Confusion Matrix, F1-Score, and Accuracy to measure performance.

## 🚀 How to Run
1. Ensure you have Python installed.
2. Install requirements: `pip install scikit-learn`
3. Run the script: `python knn_classifier.py`
```