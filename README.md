# Machine Learning Classification & Deployment using Streamlit

## Problem Statement
The objective of this project is to implement multiple machine learning classification models, compare their performance using standard evaluation metrics, and deploy them through an interactive Streamlit web application.

---

## Dataset Description
The Wine Quality dataset contains physicochemical attributes of wine samples.  
The goal is to predict the quality category of wine based on these features.

- Number of Instances: ~1600  
- Number of Features: 11  
- Target Variable: Wine Quality (multi-class classification)

---

## Models Implemented
The following models were trained and evaluated on the same dataset:

1. Logistic Regression  
2. Decision Tree  
3. K-Nearest Neighbors  
4. Naive Bayes  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

---

## Evaluation Metrics
For each model, the following metrics were computed:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

---

## Model Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|------|------|------|------|------|------|------|
| Logistic Regression | 0.5435 | 0.8021 | 0.5293 | 0.5435 | 0.5122 | 0.2736 |
| Decision Tree | 0.9229 | 0.9509 | 0.9232 | 0.9229 | 0.9230 | 0.8853 |
| KNN | 0.6718 | 0.9173 | 0.6677 | 0.6718 | 0.6624 | 0.4962 |
| Naive Bayes | 0.3457 | 0.6896 | 0.4417 | 0.3457 | 0.3818 | 0.1294 |
| Random Forest | 0.9380 | 0.9917 | 0.9390 | 0.9380 | 0.9377 | 0.9071 |
| XGBoost | 0.9241 | 0.9830 | 0.9246 | 0.9241 | 0.9240 | 0.8864 |

---

## Observations

**Logistic Regression:**  
Provides a simple linear baseline but underfits complex relationships.

**Decision Tree:**  
Captures nonlinear patterns effectively and performs strongly.

**KNN:**  
Gives moderate performance and is sensitive to feature scaling.

**Naive Bayes:**  
Lower performance due to independence assumptions among features.

**Random Forest:**  
Best performing model with highest accuracy and MCC due to ensemble averaging.

**XGBoost:**  
Very competitive with excellent AUC, slightly below Random Forest.

---

## Streamlit Application Features

- Upload custom test dataset (CSV)
- Select model from dropdown
- Automatic preprocessing
- Display evaluation metrics
- Confusion matrix visualization

---

## Project Structure
streamlit_app.py
models.py
requirements.txt
README.md
*.pkl files


---

## How to Run Locally

pip install -r requirements.txt
streamlit run streamlit_app.py