# 📊 Customer Churn Prediction Pipeline

A complete end-to-end machine learning pipeline to predict telecom customer churn using Python, scikit-learn, and SMOTE. The project handles real-world imbalanced data and is ready for deployment or integration in apps or APIs.

---

## 📌 Project Overview

Customer churn is a critical business problem in telecom and subscription-based industries. This project builds a robust machine learning pipeline to predict if a customer is likely to churn based on historical behavior and subscription details.

---

## 🚀 Features

- Data cleaning and preprocessing
- Label encoding and one-hot encoding
- Feature scaling using StandardScaler
- Handling imbalanced data using **SMOTE**
- Model training with:
  - Logistic Regression (threshold-tuned)
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
- Model evaluation (recall-focused)
- Exporting pipeline and encoders for deployment

---

## 📂 Dataset

The dataset is a telecom customer dataset containing features like:
- Demographics (gender, senior citizen)
- Account info (tenure, contract type)
- Service usage (internet service, tech support)
- Target: `Churn` (Yes/No)

> Data file: `es.csv` (not included here for licensing/privacy reasons)

---

## 🧰 Tech Stack

- Python 3.x
- pandas & numpy
- scikit-learn
- imbalanced-learn (SMOTE)
- joblib

---

## 🧠 Models & Evaluation

| Model               | Recall (Churn) | Precision (Churn) | Notes                                                  |
|---------------------|----------------|-------------------|--------------------------------------------------------|
| Logistic Regression | **94%**        | 41%               | High recall after lowering threshold to 0.4            |
| KNN                 | 93%            | 68%               | Balanced performance                                   |
| SVM                 | N/A            | N/A               | Comparable, tuned with `C=3.0` and class weights       |

> Final model: **Logistic Regression**, due to its high recall, which is critical in churn prediction tasks.

---

## ⚙️ How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/churn-prediction-pipeline.git
   cd churn-prediction-pipeline
