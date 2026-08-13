# 🛍️ Retail Intelligence Dashboard with Explainable AI
### AI-Enabled Business Analytics | Explainable AI | RAG-Powered Insights

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)
![RAG](https://img.shields.io/badge/LLM-RAG-purple)
![Status](https://img.shields.io/badge/Status-Working%20Prototype-brightgreen)

---

# 🧠 Project Overview

An AI-powered Retail Intelligence Dashboard that turns raw retail transaction data into explainable, actionable business insights — combining classical ML, Explainable AI (SHAP), and an LLM layer (Gemini) for natural-language insights and Retrieval-Augmented Generation (RAG).

Originally developed as an academic extension of my Business Analytics internship at Maincrafts Technology, this project grew from a notebook-based EDA/ML pipeline into a working Streamlit application with LLM-assisted dataset generalization — so it isn't limited to one fixed dataset schema.

**This is a working prototype, not a finished product.** The core ML pipeline (forecasting, segmentation, anomaly detection, SHAP) runs entirely independently of any LLM and is fully reliable. The AI-generated insights and RAG chatbot depend on Gemini's free-tier API, which has a genuinely small daily request quota — the app is built to degrade gracefully (clear fallback messages, no crashes) when that quota is exhausted, rather than pretending otherwise.

---

# 🎯 Problem Statement

Traditional dashboards answer: **What happened?**

This project aims to also answer:

- Why did it happen? *(Explainable AI via SHAP)*
- What will happen next? *(Forecasting)*
- Which customers/transactions need attention? *(Segmentation + anomaly detection)*
- Can I just ask a question instead of building a report? *(RAG-powered chat)*
- Does this work on data that isn't Superstore specifically? *(LLM-assisted column mapping)*

---

# 📦 Project Modules

| Module | Description | Status |
|---------|-------------|--------|
| 01 — Dataset Upload & Column Mapping | Upload CSV/XLSX; LLM auto-detects and maps columns to expected roles, with a clarification step for uncertain matches | ✅ Complete |
| 02 — Automatic EDA | Statistical summaries, distributions, correlations | ✅ Complete |
| 03 — Business KPIs | Sales, Profit, Orders, Customers, Region breakdown | ✅ Complete |
| 04 — Sales Forecasting | Random Forest on aggregated monthly data | ✅ Complete |
| 05 — Customer Segmentation | KMeans on RFM features (Recency, Frequency, Monetary) | ✅ Complete |
| 06 — Anomaly Detection | Isolation Forest — flags both deep-loss and high-profit outlier transactions | ✅ Complete |
| 07 — Explainable AI | SHAP on both the forecasting model and the anomaly detector, with a plain-language summary and a technical toggle | ✅ Complete |
| 08 — AI Business Insights | Gemini-generated "what's working / what needs attention" summary, grounded in real computed findings | ✅ Complete |
| 09 — RAG Chat | Ask questions about your dashboard; answers retrieved from a knowledge base built dynamically from your actual data (not hardcoded) | ✅ Complete |
| 10 — Budget Intelligence *(future)* | Discount-safety thresholds, segment-based budget allocation simulator | ⏳ Planned |
| Delivery Estimator *(future)* | Predicted delivery windows from historical ship-date patterns | ⏳ Planned |

---

# 🛠 Tech Stack

## Data & ML
- Python, Pandas, NumPy
- Scikit-learn — Random Forest (forecasting), KMeans (segmentation), Isolation Forest (anomaly detection)
- SHAP — explainability for both the forecasting model and the anomaly detector

## Visualization
- Plotly (interactive charts throughout the dashboard)

## LLM / AI Layer
- Google Gemini API (`gemini-flash-latest` for generation, `gemini-embedding-001` for embeddings)
- Custom RAG pipeline — embeddings + cosine similarity retrieval over a dynamically built knowledge base (no LangChain/vector-DB dependency; built directly with `google-genai` + NumPy)
- LLM-assisted column mapping for dataset generalization, with timeout + fuzzy-match fallback if the LLM is unavailable

## App
- Streamlit, with a custom light theme and cache-based session state to avoid redundant LLM calls across reruns

---

# 📊 Dataset

Developed and validated against the **Sample Superstore dataset** (Kaggle) — 9,994 transactions, 21 columns (order date, ship date, customer, sales, profit, discount, quantity, region, category, etc.) — but the column-mapping layer is designed to generalize to other retail transactional datasets with a similar shape.

---

# 🔍 Key Findings (Sample Superstore dataset)

- Discount-profitability breakeven sits around **20%** — discounts beyond that point are reliably associated with negative profit, confirmed independently through EDA, anomaly detection, *and* SHAP.
- RFM segmentation (KMeans, k=4) identifies four clear customer groups: **Champions, Loyal High-Value, At-Risk, and Lost/Churned** — each with distinct recency/frequency/monetary profiles.
- Isolation Forest flags ~5% of transactions as anomalous, splitting cleanly into two tails: **deep-loss** (high discount + large negative profit) and **high-profit** (zero discount, large B2B orders) — both are legitimate patterns, not data errors.
- SHAP shows **Month/seasonality** as the strongest driver of the sales forecast, and **Quantity + Discount** as the strongest drivers behind flagged anomalies.
- The forecasting model's R² (~0.11–0.13) is modest — documented honestly as a finding: transaction-level retail sales forecasting needs richer external signals (promotions, macro data) to move beyond directional accuracy.

---

# 📂 Repository Structure

```text
Retail-Intelligence-Dashboard-XAI/
│
├── app.py                     # Streamlit entrypoint and full app flow
├── requirements.txt
├── .env.example                # Template for GEMINI_API_KEY (never commit the real .env)
├── Sample - Superstore.csv     # Reference dataset
│
├── utils/
│   ├── data_loader.py          # CSV/XLSX loading
│   ├── column_mapping.py       # LLM-assisted column role detection + fallback
│   ├── analytics.py            # RFM, forecasting, anomaly detection, SHAP
│   ├── llm.py                  # Gemini client + insight generation (timeout-protected)
│   ├── rag.py                  # Knowledge base construction, embeddings, retrieval, chat
│   └── ui.py
│
├── About-RAG/, ML-04-Isolation-Forest.md, Retail-AI-03-*.md
│                                # Earlier learning-journey notes from the RAG/ML build process
│
└── README.md
```

---

# ⚠️ Known Limitations (Honest Section)

- **Gemini free-tier quota is small** — AI insights and chat may show a graceful fallback message under heavy use rather than a live response. The rest of the dashboard is unaffected.
- **Column-mapping is wired into the upload flow**, but not yet retrofitted across every internal module — some analytics functions still expect fairly standard column names under the hood. A full retrofit is on the roadmap.
- **RAG answers are limited to precomputed findings** in the knowledge base — it can summarize and explain what's already been calculated (region sales, segments, anomalies, trends), but can't yet perform new on-the-fly calculations beyond that. Extending this to agentic tool-calling is a planned next step.
- **Not a live/real-time system** — works on a single uploaded snapshot per session, not streaming or connected data sources.

---

# 🚀 Future Roadmap

- **Module 10 — Budget Intelligence:** discount-safety thresholds, segment-based retention budget allocation, a live what-if discount simulator
- **Delivery Estimator:** predicted delivery windows from historical ship-date patterns by region/ship mode
- **Agentic RAG:** move from fixed knowledge-base retrieval to LLM tool-calling, so the chat can compute new answers on demand instead of only summarizing precomputed findings
- **Full column-mapping retrofit** across every analytics module, not just the upload step
- Deploy a public demo instance once a higher-tier LLM quota is in place

---

# 🎓 Academic Context

**Student:** Nidhi Sharma
**Program:** B.Tech Computer Science & Engineering (AI & ML), Minor in Data Science & Machine Learning
**College:** Jawaharlal Nehru Government Engineering College (JNGEC), Sundernagar
**Internship:** Maincrafts Technology — Business Analytics

---

# 🌱 Learning Philosophy

This repository reflects an approach to learning AI through implementation rather than theory alone. Every module here — from RFM math to SHAP interpretation to debugging LLM rate limits at inconvenient hours — was built, broken, and fixed in the process of actually shipping something. The commit history is the honest version of the story.

---

## ⭐ Status

**Working prototype.** Core ML pipeline (Modules 1–7) is fully functional and independent of any external API. AI insights and RAG chat (Modules 8–9) depend on Gemini API availability. Actively maintained as time allows.
