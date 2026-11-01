# Healthcare Patient Risk Stratification System

An end-to-end patient risk-stratification pipeline built to predict healthcare risks and analyze cost metrics. This project combines a machine learning backend, an interactive web dashboard, and a parallel business intelligence analysis in Excel.

## 🚀 Features & Components

### 1. Machine Learning Pipeline (Python)
* **Data Engineering:** Cleaned, preprocessed, and joined 4 separate relational healthcare datasets containing over 200+ patient records using **Pandas**.
* **Feature Engineering:** Extracted relevant health indicators and prepared datasets for predictive modeling.
* **Predictive Classifier:** Trained a **Logistic Regression** model using **scikit-learn** to classify patient risk tiers.
* **Evaluation Metrics:** Validated model performance comprehensively using Precision, Recall, F1-Score, and ROC-AUC curves.

### 2. Interactive Web Application (Streamlit)
* Developed a user-friendly, real-time dashboard using **Streamlit**.
* Allows clinicians or administrators to input patient parameters and instantly receive live risk scores and classification outputs.

### 3. Business Intelligence Analysis (Excel)
* Built a parallel, standalone spreadsheet model for rapid clinical cost-by-segment reporting.
* Leveraged dynamic formulas like `VLOOKUP` to map data across tables.
* Implemented **PivotTables** and dynamic **PivotCharts** to break down financial data by patient risk segments.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Data Libraries:** Pandas, NumPy
* **Machine Learning:** scikit-learn
* **Deployment/UI:** Streamlit
* **Analytics & Visualization:** Microsoft Excel (VLOOKUP, PivotTables, PivotCharts)

---

## 📂 Repository Structure

* `Healthcare_risk/`
  * `patients.csv`, `labs.csv`, `outcomes.csv` — Raw relational healthcare datasets.
  * `risk_model1.pkl` — The trained Logistic Regression model file.
  * `roc_curve.png` — Model evaluation visualization.
