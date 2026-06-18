# ML Homework – Stellar Classification & Heart Attack Risk Prediction

This repository contains my work for a machine learning assignment covering two classification tasks:

- **Question 1 – Astronomical Classification**  
  Classify objects as *galaxy*, *star*, or *quasar* using spectral features from the SDSS dataset.  
  *Models:* SVM (linear kernel, C tuning), Logistic Regression (with coefficient interpretation), evaluation with ROC, confusion matrices.

- **Question 2 – Early Heart Attack Risk Detection**  
  Predict risk of heart attack from patient health records.  
  *Models:* SVM (RBF, grid search for C/gamma), Perceptron (custom implementation + scikit‑learn), MLP (GridSearchCV), Decision Trees, and a full comparison.

## Files
- `stellar_classification_heart_risk.ipynb` – Full Jupyter Notebook with step‑by‑step code for both questions.
- `star.csv` – Dataset for Question 1.
- `heart_attack_risk.csv` – Dataset for Question 2.

## How to Run
1. Clone the repository or download the files.
2. Open the notebook in Jupyter, VS Code, or Google Colab.
3. Install required packages: `pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn`
4. Run all cells sequentially.

## Why this is public
I’m sharing this as part of my learning portfolio to demonstrate end‑to‑end ML pipeline skills: data cleaning, feature encoding, handling imbalance, model training, hyperparameter tuning, and evaluation.

*Assignment date: [Month/Year]*  
*Course: [Course Name if you want]*
