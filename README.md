# 🏦 Loan Approval Prediction — End-to-End Machine Learning Workflow

An end-to-end Machine Learning project that automates loan eligibility evaluation to assist financial institutions in managing credit risk and accelerating loan decisions.

---

## 📌 Problem Statement
Manual loan evaluation is time-consuming and prone to subjective bias. This project builds a binary classification model to predict whether an applicant will be approved (`1`) or rejected (`0`) based on financial, demographic, and credit profile attributes.

---

## 📊 Dataset Overview
The dataset contains applicant financial profiles with features including:
* **Demographics & Profile:** `Age`, `Gender`, `Education`, `Home Onwership`, `Employee Experience`
* **Financial Metrics:** `Person Income`, `Loan Amount`, `Loan Intent`, `Loan interest Rate`, `Loan percentage`
* **Credit History:** `Credit History`, `Credit Score`, `Previous Loan`
* **Target Label:** `Loan Status` (Approved / Rejected)

---

## 🛠️ Data Preprocessing & Pipeline
1. **Data Cleaning:** Checked for duplicate records and filled missing values using numerical median strategy.
2. **Feature Engineering & Encoding:** Applied One-Hot Encoding (`pd.get_dummies`) to convert categorical features (`Gender`, `Education`, `Home Onwership`, `Loan Intent`, `Previous Loan`) into machine-readable numeric formats.
3. **Data Splitting:** Divided dataset into 80% Training set and 20% Testing set using stratified sampling to preserve class balances.

---

## ⚙️ Model & Evaluation
A **Logistic Regression** baseline classifier was trained to evaluate approval probabilities.

### **Evaluation Metrics:**
* **Accuracy:** ~81.0%
* **Precision:** ~0.79
* **Recall:** ~0.98
* **F1-Score:** ~0.87
* **ROC-AUC Score:** ~0.80

> **Key Insight:** `Credit History` and `Credit Score` were identified as the strongest predictive indicators for loan qualification.

---

## 💻 Web Application (Gradio)
The project includes an interactive web interface built using **Gradio**, allowing users to input applicant features and receive real-time approval predictions with confidence probabilities.

---

## ⚠️ Limitations & Future Work
* **Linear Decision Boundary:** Logistic Regression assumes linear separability; complex non-linear patterns may be missed.
* **Future Improvements:** Experiment with ensemble algorithms (Random Forest, XGBoost) and implement feature scaling (`StandardScaler`) to enhance performance.
