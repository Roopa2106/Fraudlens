# 🛡️ FraudLens: Hybrid Fraud Detection System
**Mini Hackathon 2026 · Problem 3: Fraud Detection System**

FraudLens flags suspicious financial transactions with two methods. **Explainable business rules** catch known fraud patterns, and an **Isolation Forest** (unsupervised ML) catches unusual behaviour that no rule describes. The two are blended into one **0–100 risk score**, and every flag comes with a plain-English reason.

---

## 1. Problem understanding
- Fraud is **rare** (usually under 2% of transactions) and often **unlabelled**, so a plain supervised classifier is not always possible.
- Fraud is about **behaviour relative to the customer**. ₹40,000 is normal for one person and alarming for another.
- Analysts need to know **why** a transaction is flagged, not only that it was.
- **Precision matters too.** Too many false alarms block genuine customers and overload the review team.

## 2. Approach

```
Raw CSV ──► Cleaning ──► Feature engineering ──► ┌─ 7 business rules ─► rule points ─┐
(any columns,           (per-customer behaviour) │                                   ├─► Risk score 0–100 ─► Dashboard
 mapped in UI)                                   └─ Isolation Forest ─► anomaly score┘   + reasons
```

### 2.1 Data cleaning
Coerce amounts to numbers, drop missing or negative amounts, parse timestamps, and sort by time.

### 2.2 Feature engineering (per customer)
| Feature | Meaning |
|---|---|
| `amount_zscore` | How many std devs this amount is above the customer's own average |
| `amount_to_avg_ratio` | Amount ÷ the customer's average spend |
| `hour`, `day_of_week` | Timing |
| `txn_in_window` | Rolling count of the customer's transactions in the last N minutes (velocity) |
| `mins_since_prev` | Gap since the customer's previous transaction |
| `location_share`, `device_share` | How often this customer uses this location or device (low = new) |

### 2.3 Rule engine (thresholds adjustable in the sidebar)
| Rule | Logic | Points |
|---|---|---|
| R1 Amount spike | z-score > 3 | 25 |
| R2 High value | amount > ₹50,000 | 15 |
| R3 Odd hour | 00:00–05:00 | 10 |
| R4 Velocity | ≥ 3 txns within 60 min | 20 |
| R5 New location | location share < 15% (customer has ≥ 5 txns) | 20 |
| R6 New device | device share < 15% | 10 |
| R7 Card testing | micro payment (< ₹50) followed by a large one within 30 min | 20 |

### 2.4 Isolation Forest
- Trained on the engineered features, with log-transformed amount and standard scaling.
- Isolates points with random splits: **anomalies need fewer splits to isolate**, so they get a higher score.
- Needs **no labels**. The contamination setting is the expected share of fraud.

### 2.5 Hybrid risk score
`risk = 100 × [(1 − w) × rule_score + w × ML_anomaly_score]`, with `w` = ML weight (default 0.5).
Levels: Low < 30 ≤ Medium < cutoff (50) ≤ High → flagged as **suspicious**.

## 3. Results (sample data: 10,965 transactions, ~2% fraud)
| Approach | Precision | Recall | F1 |
|---|---|---|---|
| Rules only | 0.76 | 0.72 | 0.74 |
| **Hybrid (w = 0.5)** | **0.77** | **0.72** | **0.74** |
| ML only | 0.49 | 0.80 | 0.61 |

ROC-AUC of the hybrid score is **0.988**. ML alone catches more fraud but creates roughly twice the false alarms. The rules add precision and explanations, and the hybrid score ranks best overall. *(Labels are used only to evaluate. Both detectors are unsupervised.)*

## 4. Dashboard features
1. **Upload any CSV** and map its columns. Only `amount` is required, and rules switch on automatically based on the columns available.
2. **KPIs**: transactions analysed, number flagged, amount at risk, average risk.
3. **🚩 Suspicious transactions**: ranked table with risk bar and "why flagged" reasons, downloadable as CSV.
4. **📊 Patterns & insights**: risk distribution, rule frequency, flag rate by hour and category, deviation-vs-anomaly scatter, auto-written insights.
5. **🔍 Investigate**: per-customer transaction timeline with flagged points highlighted.
6. **✅ Model evaluation** (if labels exist): precision, recall, F1, ROC-AUC, confusion matrix, rules vs ML vs hybrid comparison.
7. **Live tuning**: change thresholds or the ML weight and every view updates instantly.

## 5. Technologies & tools
| Tool | Use |
|---|---|
| Python 3.10+ | Core language |
| pandas, NumPy | Cleaning, feature engineering |
| scikit-learn | Isolation Forest, StandardScaler, metrics |
| Plotly | Interactive charts |
| Streamlit | Web dashboard |
| GitHub | Version control & submission |
| ChatGPT / Claude | Coding assistance (all logic understood and explained by the team) |

## 6. How to run
```bash
pip install -r requirements.txt
python generate_data.py        # optional: writes data/transactions.csv
streamlit run app.py
```
In the sidebar, choose **Use sample data**, or **Upload CSV** and map the columns.

**Public datasets that work directly:** Kaggle *Credit Card Fraud Detection* (map `Amount`, `Class`), *PaySim* (`amount`, `nameOrig`, `step`, `type`, `isFraud`), and *Credit Card Transactions Fraud* (`amt`, `cc_num`, `trans_date_trans_time`, `category`, `city`, `is_fraud`).

## 7. Project structure
```
fraud-detection/
├── app.py             # Streamlit dashboard (UI)
├── fraud_engine.py    # Features, rules, Isolation Forest, risk score
├── generate_data.py   # Synthetic dataset with injected fraud patterns
├── requirements.txt
└── README.md
```

## 8. Team roles (4 members)
| Member | Owns | Explains to judges |
|---|---|---|
| **M1: Data & features** | Loading, cleaning, `engineer_features()`, sample data | Why per-customer z-scores and velocity features matter |
| **M2: Rules engine** | `apply_rules()`, thresholds, reason text | Each rule, its business logic and weight |
| **M3: ML model** | `apply_isolation_forest()`, hybrid score, evaluation | How Isolation Forest works, contamination, precision/recall trade-off |
| **M4: Dashboard & presentation** | `app.py`, charts, insights, slides, demo | User flow, live demo, business impact |

## 9. 2-hour plan
| Time | Activity | Output |
|---|---|---|
| 0–10 | Briefing | Problem understood |
| 10–20 | Planning | Roles split, dataset columns checked, repo created |
| 20–65 | **Phase 1** | Cleaning + features (M1), 4 rules (M2), Isolation Forest (M3), basic Streamlit table (M4) |
| 65–75 | **Midway review** | Show the pipeline diagram + first flagged list |
| 75–105 | **Phase 2** | Remaining rules, hybrid score, charts, evaluation tab, insights, slides |
| 105–120 | **Final demo** | 5-minute presentation + live demo + Q&A |

## 10. Limitations & future work
- Thresholds are hand-set. With labelled history, tune them or train a supervised model (XGBoost) with SMOTE or class weights.
- Batch scoring only. Next step: real-time scoring through an API (FastAPI) and Kafka stream.
- Add network features (shared devices or IPs across accounts) and a feedback loop where analyst decisions retrain the model.
- SHAP explanations for the ML part.
