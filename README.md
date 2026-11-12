# 🛒 CartSense AI – Enterprise Customer Purchase Intention Prediction Platform

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E.svg)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**CartSense AI** is a production-grade machine learning SaaS application that predicts whether an e-commerce visitor is likely to complete a purchase based on their browsing behavior and real-time session telemetry.

Built on the **Online Shoppers Purchasing Intention dataset** from the **UCI Machine Learning Repository** (12,330 customer sessions), CartSense AI delivers a high-energy, colorful, glassmorphic analytics interface equipped with real-world enterprise capabilities.

---

## 🌟 What Makes CartSense AI a "Real App"?

1. **Top SaaS Navigation Bar:** Integrated platform telemetry with a live animated pulse indicator (`● ML Cluster Active`), real-time latency monitoring (`11ms`), model versioning (`Random Forest`), and dataset health status.
2. **Distinct Signature Color KPI Cards:** High-energy glassmorphic cards with individual neon gradients and glowing hover accents:
   - 🟣 **Purple / Violet Gradient:** Total Shopping Sessions
   - 🟢 **Emerald / Mint Gradient:** Completed Purchases & Conversion Lift
   - 🔴 **Coral / Rose Gradient:** Abandoned Sessions & Drop-off Risk
   - 🔵 **Electric Azure Gradient:** Overall Store Conversion Rate
   - 🟡 **Golden Amber Gradient:** Google Page Values & Dwell Metrics
3. **Interactive Session Benchmark Radar Chart:** A 6-axis Spider/Radar chart plotting the active visitor session against an **"Average Buyer"** benchmark and **"Casual Browser"** benchmark in real time.
4. **Automated E-Commerce Action Engine:** Real applications don't just output a probability—they recommend actions:
   - *Purchase Likely:* Auto-triggers **Express 1-Click Checkout**, **Cart Reservation Timer (15 mins)**, and **One-Click Add-on Offers**.
   - *Purchase Unlikely:* Auto-triggers **Exit-Intent Retention 10% Vouchers**, **Social Proof Carousels**, and **Proactive Live Chat Assistance**.
5. **Enterprise Batch CSV Scoring:**
   - Upload any `.csv` file containing customer sessions.
   - Includes a 1-click **"Download Sample Batch CSV Template"** (25 sessions).
   - Generates instant batch predictions, probability scores, summary metrics, and a downloadable scored CSV.
6. **Decision Threshold Sensitivity Simulator (What-If Analysis):**
   - Interactive probability slider $T \in [0.10, 0.90]$ that dynamically recomputes Precision, Recall, F1, and the Confusion Matrix in real time.
7. **Multi-Model ROC Curves & Sunburst Funnel:**
   - Direct ROC curve comparison for Logistic Regression, Decision Tree, and Random Forest.
   - Multi-level interactive Sunburst chart: `VisitorType` $\rightarrow$ `Day Type` $\rightarrow$ `Purchase Outcome`.

---

## 📊 Dashboard Modules

| Page | Module | Key Real-App Features |
|---|---|---|
| **1. 📊 Overview Dashboard** | Macro Retail Analytics | Multi-color KPI cards, Donut class imbalance chart, monthly volume vs conversion velocity, Google Page Value violin spread |
| **2. 🔮 Live Purchase Prediction** | Dual-Tab Inference Hub | **Tab A:** 17 inputs, 1-click presets, probability gauge, likelihood glow banner, 6-axis radar benchmark, CartSense action engine.<br>**Tab B:** Enterprise Batch CSV upload, sample template generator, scored CSV export |
| **3. 📈 Customer Analytics** | Segmentation & Friction | Interactive 3-level Sunburst chart, acquisition channel rankings (Turbo gradient), bounce vs exit scatter plot, catalog dwell histogram, weekend conversion lift |
| **4. 🤖 Machine Learning Models** | Multi-Model Benchmark | Comparative performance matrix, grouped bar chart, multi-model ROC curves, and **Interactive Threshold Simulator** with live confusion matrix updates |
| **5. 🎯 Feature Importance** | Explainable AI | Aggregated Gini importance chart (electric cyan gradient), top 15 encoded feature node weights, behavioral factor deep-dive cards, statistical pattern disclaimer |
| **6. 📁 Dataset Explorer** | Query & Audit Suite | Multi-filter query engine (month, visitor type, outcome, weekend), data summary statistics, 100% data hygiene audit, filtered CSV download |

---

## 🛠️ Technology Stack

- **Core Programming:** Python 3.12
- **Web Interface:** Streamlit (Glassmorphic Vibrant Theme)
- **Data Engineering:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (Pipelines, ColumnTransformer, Classifiers, Evaluation Metrics)
- **Interactive Visualizations:** Plotly Express & Plotly Graph Objects
- **Model Serialization:** Joblib, JSON

---

## 📁 Project Directory Structure

```text
cartsense/
│
├── .streamlit/
│   └── config.toml                  # Custom Streamlit vibrant theme configuration
│
├── data/
│   └── online_shoppers_intention.csv # UCI Machine Learning dataset (12,330 rows)
│
├── models/
│   ├── best_model.joblib            # Production Random Forest pipeline
│   ├── logistic_regression.joblib   # Trained Logistic Regression classifier
│   ├── decision_tree.joblib         # Trained Decision Tree classifier
│   ├── random_forest.joblib         # Trained Random Forest classifier
│   ├── preprocessor.joblib          # Fitted ColumnTransformer (Scalers & Encoders)
│   ├── model_metrics.json           # Test evaluation metrics for all 3 models
│   ├── feature_importance.json      # Gini importance scores & ranked features
│   ├── metadata.json                # Feature schemas & default session values
│   └── test_sample.joblib           # Holdout test sample for live matrix rendering
│
├── src/
│   ├── __init__.py                  # Package initializer
│   ├── data_loader.py               # Cached data loading & aggregations (@st.cache_data)
│   ├── ml_pipeline.py               # Live prediction, action engine & batch scoring
│   ├── style.py                     # Ultra-vibrant CSS, KPI cards, banners & badges
│   └── visualizations.py            # 15 interactive Plotly charts in vibrant neon palettes
│
├── app.py                           # Main Streamlit web application & router
├── train_model.py                   # Standalone training & evaluation pipeline
├── test_app.py                      # Automated test suite (100% passing)
├── requirements.txt                 # Production Python dependencies
├── render.yaml                      # Render Infrastructure-as-Code specification
├── .gitignore                       # Git exclusions (.venv, __pycache__, etc.)
└── README.md                        # Complete project documentation
```

---

## ⚡ Quick Start: Run Locally in VS Code

Run the following commands in your VS Code terminal inside `cartsense`:

```powershell
# 1. (Optional) Activate your virtual environment
.\.venv\Scripts\Activate.ps1

# 2. (Optional) Verify all automated tests
python test_app.py

# 3. Launch the CartSense AI web application
streamlit run app.py
```

The application will immediately launch at **`http://localhost:8501`**.

---

## 🐙 Step-by-Step GitHub Upload Instructions

```bash
git init
git add .
git commit -m "Complete CartSense AI enterprise web application"
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/cartsense.git
git push -u origin main
```

---

## ☁️ Step-by-Step Render Deployment Instructions

1. Go to [dashboard.render.com](https://dashboard.render.com/) and sign in.
2. Click **New +** and select **Web Service**.
3. Connect your GitHub repository `cartsense`.
4. Confirm the settings:
   - **Name:** `cartsense-ai`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT`
   - **Plan:** `Free`
5. Click **Create Web Service**.

Your live URL will be provisioned automatically (e.g. `https://cartsense-ai.onrender.com`).
