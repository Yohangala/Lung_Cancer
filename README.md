# 🫁 Lung Cancer Prediction using Ensemble Machine Learning

> **Published Research** · INDIACom 2025 (12th International Conference on Computing for Sustainable Global Development) · BVICAM, New Delhi · ISSN 0973-7529

[![Conference](https://img.shields.io/badge/INDIACom-2025-blue?style=flat-square)](https://bvicam.in/INDIACom/news/INDIACom%202025%20Proceedings/Main/papers/984.pdf)
[![Python](https://img.shields.io/badge/Python-3.8+-green?style=flat-square&logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-96.64%25-red?style=flat-square)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yohangala/Lung_Cancer/blob/main/LungCancer.ipynb)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://lung-cancer-detection.streamlit.app)

---

## 📄 Published Paper

**"Predicting Lung Cancer using Ensemble Machine Learning Algorithms: A Comprehensive Study"**

*Kartik Deshmukh · Yohan Gala · Shubham Darji · Pradnya Patil*  
*K J Somaiya Institute of Technology, Sion, Mumbai, India*

📎 [Read the Full Paper (PDF)](https://bvicam.in/INDIACom/news/INDIACom%202025%20Proceedings/Main/papers/984.pdf)

---

## 🎯 Problem Statement

Lung cancer is the leading cause of cancer-related deaths worldwide. Symptoms often appear only in later stages, reducing the five-year survival rate to ~10%. Early detection can push that survival rate above 50%. This project builds a machine learning pipeline that predicts lung cancer risk from clinical and symptomatic survey data — offering a low-cost, non-invasive diagnostic support tool.

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score | Cross-Val |
|---|---|---|---|---|---|
| **XGBoost** 🏆 | **96.64%** | **0.97** | **0.97** | **0.97** | **0.9434** |
| KNN | 95.80% | 0.96 | 0.96 | 0.96 | 0.9099 |
| SVC | 95.80% | 0.96 | 0.96 | 0.96 | 0.9454 |
| Random Forest | 95.80% | 0.96 | 0.96 | 0.96 | 0.9392 |
| MLP Classifier | 95.80% | 0.96 | 0.96 | 0.96 | 0.9497 |
| Ensemble Voting | 95.80% | 0.96 | 0.96 | 0.96 | 0.9413 |
| Gradient Boosting | 94.96% | 0.95 | 0.95 | 0.95 | 0.9434 |
| Logistic Regression | 94.12% | 0.94 | 0.94 | 0.94 | 0.9034 |
| Decision Tree | 90.76% | 0.91 | 0.91 | 0.91 | 0.9307 |
| Gaussian Naive Bayes | 90.76% | 0.91 | 0.91 | 0.91 | 0.8719 |
| Multinomial Naive Bayes | 75.63% | 0.76 | 0.76 | 0.76 | 0.7542 |

> **XGBoost achieved 96.64% accuracy** with a cross-validation score of 0.9434, making it the best-performing individual classifier. The Ensemble Voting Classifier (combining all 10 models via soft voting) achieved 95.80% accuracy with a cross-validation of 0.9413.

---

## 🏗️ Pipeline Architecture

```
Raw Survey Data (309 samples, 16 features)
        │
        ▼
┌─────────────────────────┐
│   Data Preprocessing    │
│  • Duplicate removal    │
│  • Label encoding       │
│  • MinMaxScaler [0,1]   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Feature Engineering   │
│  • ANXYELFIN interaction │
│    (ANXIETY × YELLOW_   │
│     FINGERS)            │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Class Balancing       │
│  • ADASYN oversampling  │
│    (focuses on hard-to- │
│    classify samples)    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  10 ML Models Trained   │
│  + Hyperparameter Tuning│
│  (RandomizedSearchCV)   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Ensemble Voting        │
│  Classifier (Soft Vote) │
└────────────┬────────────┘
             │
             ▼
    Evaluation: Accuracy, Precision,
    Recall, F1, 5-Fold Cross-Validation
```

---

## 📁 Dataset

- **Source:** [Lung Cancer Survey Dataset — Kaggle](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer)
- **Samples:** 309 observations
- **Features:** 16 columns (15 clinical/symptomatic features + 1 binary target)
- **Target:** `LUNG_CANCER` — YES (cancer present) / NO (cancer absent)

> **Note:** Download `lung_cancer_survey.csv` from [Kaggle](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer) and place it in the repo root before running the notebook or app.

### Features

| Feature | Description | Type |
|---|---|---|
| GENDER | Patient's gender | Categorical |
| AGE | Patient's age in years | Numerical |
| SMOKING | Smoking status (0/1) | Numerical |
| YELLOW_FINGERS | Yellow fingers present (0/1) | Numerical |
| ANXIETY | Anxiety present (0/1) | Numerical |
| PEER_PRESSURE | Peer pressure (0/1) | Numerical |
| CHRONIC DISEASE | Chronic disease (0/1) | Numerical |
| FATIGUE | Experiencing fatigue (0/1) | Numerical |
| ALLERGY | Allergies present (0/1) | Numerical |
| WHEEZING | Wheezing present (0/1) | Numerical |
| ALCOHOL CONSUMING | Alcohol consumption (0/1) | Numerical |
| COUGHING | Coughing (0/1) | Numerical |
| SHORTNESS OF BREATH | Shortness of breath (0/1) | Numerical |
| SWALLOWING DIFFICULTY | Swallowing difficulty (0/1) | Numerical |
| CHEST PAIN | Chest pain (0/1) | Numerical |
| **LUNG_CANCER** | **Target: cancer present (0/1)** | **Categorical** |

---

## 🧠 Models Implemented

1. **Logistic Regression** — interpretable baseline
2. **Decision Tree** — nonlinear, interpretable boundaries
3. **K-Nearest Neighbors (KNN)** — distance-based classification
4. **Gaussian Naive Bayes** — probabilistic, handles continuous features
5. **Multinomial Naive Bayes** — suited for count/categorical data
6. **Support Vector Classifier (SVC)** — RBF kernel, high-dimensional
7. **Random Forest** — bagging ensemble of decision trees
8. **Gradient Boosting** — sequential boosting of weak learners
9. **XGBoost** — optimized gradient boosting ← **best performer**
10. **MLP Classifier** — neural network for nonlinear patterns
11. **Ensemble Voting Classifier** — soft-vote average of all 10 models

All models tuned with **RandomizedSearchCV** (5-fold cross-validation).

---

## 🔑 Key Techniques

- **ADASYN** (Adaptive Synthetic Sampling): generates synthetic minority-class samples, weighted toward harder-to-classify cases — ensures the model learns to detect actual lung cancer cases, not just the majority class
- **Interaction Feature Engineering**: `ANXYELFIN = ANXIETY × YELLOW_FINGERS` — captures synergistic symptom effects
- **Soft Voting**: ensemble combines predicted probabilities from all 10 models, not just hard class labels — more stable predictions
- **5-Fold Cross-Validation**: prevents overfitting to any single train/test split

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn jupyter streamlit
```

### Run the Notebook

```bash
# Clone the repo
git clone https://github.com/Yohangala/Lung_Cancer.git
cd Lung_Cancer

# Launch Jupyter
jupyter notebook "LungCancer.ipynb"
```

> Make sure `lung_cancer_survey.csv` is in the same directory as the notebook.

### Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📂 Repository Structure

```
Lung_Cancer/
├── LungCancer.ipynb          # Main notebook — full pipeline
├── lung_cancer_survey.csv    # Dataset (309 samples, 16 features) — download from Kaggle
├── app.py                    # Streamlit demo app
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@inproceedings{deshmukh2025lungcancer,
  title     = {Predicting Lung Cancer using Ensemble Machine Learning Algorithms: A Comprehensive Study},
  author    = {Deshmukh, Kartik and Gala, Yohan and Darji, Shubham and Patil, Pradnya},
  booktitle = {Proceedings of the 12th International Conference on Computing for Sustainable Global Development (INDIACom)},
  year      = {2025},
  publisher = {BVICAM, New Delhi},
  issn      = {0973-7529},
  isbn      = {978-93-80544-58-8},
  pages     = {1049--1054}
}
```

---

## 👥 Authors

| Name | Institution | Email |
|---|---|---|
| Kartik Deshmukh | K J Somaiya Institute of Technology | kartik.sd@somaiya.edu |
| **Yohan Gala** | K J Somaiya Institute of Technology | yohan.gala@somaiya.edu |
| Shubham Darji | K J Somaiya Institute of Technology | shubham.darji@somaiya.edu |
| Pradnya Patil | K J Somaiya Institute of Technology | pradnya08@somaiya.edu |

---

## 🔮 Future Work

- Integrate imaging data (CT scans) with clinical data for multimodal prediction
- Apply SHAP / LIME for model interpretability and clinical explainability
- Expand dataset size and explore deep learning architectures
- Incorporate multi-omics data (genomic, proteomic) for holistic risk models

---

*This research was presented at INDIACom 2025 and published in the conference proceedings (ISSN 0973-7529).*
