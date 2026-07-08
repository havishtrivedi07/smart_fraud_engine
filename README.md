# 🛡️ Smart Fraud Risk Detection & Auditing Center

A real-time credit card fraud detection system built for the Microsoft Summer of Code Hackathon. The project combines machine learning with explainable AI to help analysts detect suspicious transactions while understanding why a transaction was flagged. Along with fraud prediction, the dashboard provides interactive visualizations, risk simulations, and graph-based fraud ring analysis to make investigations faster and more transparent.

---

## 👥 Team Members

* Havish – Machine Learning, Feature Engineering & Model Calibration
* Darshil – Frontend Development & Streaming Integration

---
### Deployement Link on steamlit : https://smartfraudengine.streamlit.app/ 

## YT demo video : https://youtu.be/4wdmhvMtYks?si=xozCHcZSXjx0VmTX
## ppt for complete explanation : https://docs.google.com/presentation/d/1pcLLJMQDyhwGaVlFVqlhUxkyR3xDJLORLJT1sB988XE/edit?slide=id.p1#slide=id.p1

## 📌 Problem Statement

Financial institutions process millions of digital transactions every day, making fraud detection both challenging and time-sensitive.

Most existing systems have two major limitations:

* Rule-based systems rely on fixed thresholds, which often generate too many false alarms and block legitimate customers.
* Black-box AI models can detect fraud accurately but usually don't explain their decisions, making them difficult to use in regulated financial environments.

Our goal was to build a system that not only detects fraud accurately but also explains every prediction in a way that analysts and auditors can trust.

---

## 🧠 Solution

Our system follows a simple pipeline:

Transaction Data → Feature Engineering → LightGBM Model → Probability Calibration → SHAP Explainability → Interactive Dashboard

### Fraud Detection Model

We use LightGBM because it is fast, efficient, and performs well on imbalanced fraud datasets.

To improve the reliability of predicted probabilities, we apply Platt Scaling (Logistic Regression Calibration) so that risk scores better reflect real-world fraud likelihood.

### Explainable AI

Every prediction is accompanied by SHAP values, showing exactly which features increased or decreased the fraud score.

Instead of only displaying "Fraud Risk: 92%", the dashboard explains why—for example:

* Large transaction amount
* Transaction made during late-night hours
* Low device trust score
* Unusual merchant category

This makes the model much easier to audit and trust.

### What-If Analysis

Analysts can modify transaction features using interactive sliders and instantly see how the fraud score changes.

This helps understand model behavior and identify which factors have the greatest impact on risk.

### Interactive Dashboard

The Streamlit dashboard includes:

* Real-time fraud prediction
* SHAP-based feature explanations
* Transaction risk visualization
* Fraud network graphs
* Interactive What-If simulations

---

## 🛠️ Tech Stack

| Category         | Tools                  |
| ---------------- | ---------------------- |
| Language         | Python                 |
| Machine Learning | LightGBM, Scikit-learn |
| Explainable AI   | SHAP                   |
| Frontend         | Streamlit              |
| Data Processing  | Pandas, NumPy          |
| Visualization    | Plotly                 |
| Model Storage    | Joblib                 |

---

## 📊 Dataset

We trained and evaluated the model using the Credit Card Fraud Detection Dataset from Kaggle.

Dataset:
https://www.kaggle.com/datasets/miadul/credit-card-fraud-detection-dataset

### Feature Engineering

Additional features were created to improve model performance, including:

* `is_night` – Detects late-night transactions
* `amount_log` – Log transformation of transaction amount
* `risk_flag_count` – Counts multiple suspicious indicators
* `velocity_amount_ratio` – Spending speed relative to recent activity
* `merchant_cat_amount_z` – Detects unusual spending within a merchant category

---

## 🚀 Running the Project

```bash
# Clone the repository
git clone https://github.com/havishtrivedi07/smart_fraud_engine.git
cd smart_fraud_engine

# Create virtual environment

# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py --data credit_card_fraud_10k.csv

# Launch the dashboard
streamlit run app.py
```

---

## ✨ Key Features

* Real-time fraud prediction
* Probability-calibrated risk scoring
* SHAP-based model explanations
* Interactive What-If analysis
* Plotly-powered visual analytics
* Streamlit dashboard for investigation and auditing
