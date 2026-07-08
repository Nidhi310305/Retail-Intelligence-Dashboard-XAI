# Retail Intelligence Dashboard with Explainable AI
### AI-Enabled Business Analytics | Academic Project | 7th Semester Internship Extension



![Python](https://img.shields.io/badge/Python-3.13-blue)




![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)




![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)




![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)



---

## 🧠 Project Overview

An AI-powered web dashboard that converts raw retail sales data into 
actionable business intelligence — combining traditional analytics with 
Machine Learning and Explainable AI.

Built as an academic extension of my Business Analytics internship at 
Maincrafts Technology, this project goes beyond descriptive dashboards 
to deliver predictive insights and model transparency.

---

## 🎯 Problem Statement

Traditional business dashboards answer *what happened*.
This dashboard answers *why it happened* and *what will happen next* —
making ML predictions explainable and actionable for non-technical 
business users.

---

## 📦 Modules

| Module | Description | Status |
|---|---|---|
| 01 — Dataset Upload | Upload retail CSV, validate columns |✅ Complete |
| 02 — Automatic EDA | Dataset summary, distributions, correlations | ✅ Complete |
| 03 — Business KPIs | Total Sales, Profit, Regional & Category performance | ✅ Complete |
| 04 — Sales Forecasting | Random Forest on aggregated monthly data | ✅ Complete |
| 05 — Customer Segmentation | KMeans clustering on RFM features | ✅ Complete |
| 06 — Anomaly Detection | Isolation Forest for unusual transactions | 🔄 In Progress |
| 07 — Explainable AI | SHAP values for prediction transparency | ⏳ Pending |
| 08 — Business Recommendations | Rule-based actionable insights from analysis | ⏳ Pending |

---

## 🛠️ Tech Stack

**Data Processing**
- Python 3.13
- Pandas, NumPy

**Visualization**
- Plotly, Matplotlib, Seaborn

**Machine Learning**
- Scikit-learn (Random Forest, KMeans, Isolation Forest)

**Explainability**
- SHAP (SHapley Additive exPlanations)

**Dashboard**
- Streamlit

---

## 📊 Dataset

Sample Superstore Dataset (Kaggle)
- 9,994 retail transactions
- 21 features including Sales, Profit, Discount, Region, Category
- Date range: 2014–2017

---

## 🔍 Key Findings So Far

- West region drives 31.58% of total revenue
- Technology category = 50.8% of total profit
- Every discount above 20% produces negative average profit (cross-validated via SQL + scatter analysis)
- 18.72% of all transactions generated a loss
- Sales follow strong Q4 seasonal pattern — November peaks at $352K
- Individual transaction-level forecasting shows high variance (R2 negative) — external signals required for production-grade forecasting

---

## 📁 Repository Structure
```
Retail-Intelligence-Dashboard-XAI/
│
├── data/
│   └── Sample-Superstore.csv
│
├── notebooks/
│   └── Superstore_EDA_ML.ipynb
│
├── app/
│   └── app.py (Streamlit dashboard — coming soon)
│
├── assets/
│   └── screenshots/
│
└── README.md
---
```
## 📈 Progress Log

| Date | Update |
|---|---|
| June 25-27, 2026 | Task 1 complete — Excel EDA, SQL queries, Power BI dashboard |
| June 29, 2026 | Track 2 started — Colab setup, EDA pipeline complete |
| July 4, 2026 | Sales Forecasting module attempted, limitations documented |
| July 5, 2026 | KMeans Customer Segmentation — completed |

---

## 🎓 Academic Context

**Student:** Nidhi Sharma
**Program:** B.Tech CSE (AI & ML) — 3rd Year
**College:** JNGEC Sundernagar, Himachal Pradesh
**Minor:** Data Science & ML, IIT Mandi
**Internship:** Maincrafts Technology | Business Analytics | June–August 2026

---

## 🚀 Deployment

Streamlit Cloud deployment coming upon project completion.
Live link will be updated here.

---

*Actively being built — commits reflect real learning progress.*
