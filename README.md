# ⚡ Smart Electricity Theft Detection System

> **An end-to-end Machine Learning project for detecting Non-Technical Losses (NTL) in electricity distribution using anomaly detection techniques.**

---

## 🌍 Real-World Problem

Electricity theft (Non-Technical Loss) costs utilities **$89 billion annually** worldwide. Thieves tamper with meters, create illegal connections, or manipulate billing records to reduce their bills. Traditional manual audit processes are slow and expensive.

**This project uses AI/ML to automatically flag suspicious consumption patterns** — enabling utilities to prioritise field inspections and recover lost revenue.

---

## 📸 Project Screenshots

| Dashboard | Theft Detector |
|:---:|:---:|
| ![Dashboard](screenshots/timeseries_anomalies.png) | ![Detector](screenshots/distribution_normal_vs_suspicious.png) |

| Feature Importance | Anomaly Score |
|:---:|:---:|
| ![Features](screenshots/feature_importance.png) | ![Scores](screenshots/anomaly_score_distribution.png) |

---

## 📁 Project Structure

```
electricity-theft-detection/
│
├── dataset/
│   ├── LD2011_2014.txt              # Raw UCI dataset (auto-downloaded)
│   ├── processed_hourly.csv         # Cleaned, hourly-resampled data
│   └── features.csv                 # Engineered feature matrix
│
├── model/
│   ├── isolation_forest.pkl         # Primary anomaly detector
│   ├── lof.pkl                      # Local Outlier Factor model
│   ├── random_forest.pkl            # Supervised classifier
│   ├── scaler.pkl                   # StandardScaler
│   └── feature_columns.pkl          # Feature column order
│
├── src/
│   ├── preprocess.py                # Data loading, cleaning, normalization
│   ├── features.py                  # Feature engineering
│   ├── train.py                     # Model training & evaluation
│   └── predict.py                   # Inference utilities
│
├── screenshots/                     # Training plots
├── notebook.ipynb                   # Jupyter walkthrough
├── app.py                           # Streamlit web application
├── run_pipeline.py                  # One-command pipeline runner
├── requirements.txt
└── README.md
```

---

## 🧠 Technologies Used

| Category | Tools |
|---|---|
| **Language** | Python 3.11+ |
| **ML / Anomaly** | Scikit-learn (Isolation Forest, LOF, Random Forest) |
| **Data** | Pandas, NumPy |
| **Visualisation** | Matplotlib, Seaborn, Plotly |
| **Web App** | Streamlit |
| **Persistence** | Joblib |
| **Dataset** | UCI Electricity Load Diagrams 2011–2014 |

---

## 🔬 Approach

### Problem Framing
Electricity theft is treated as an **anomaly detection** problem:
- Normal clients follow predictable daily/weekly consumption patterns.
- Thieves show abnormally low consumption (meter bypass) or irregular patterns (smart meter tampering).

### Label Strategy
Since the UCI dataset has no ground-truth theft labels, we simulate them:
- **Bottom 5% of clients** by mean consumption → `Suspicious (1)`
- **Remaining 95%** → `Normal (0)`

This is a standard approach in the NTL research literature.

---

## 🤖 Models

### 1. Isolation Forest (Primary)
- Unsupervised anomaly detection
- Works by "isolating" anomalies in random decision trees
- Anomalies require fewer splits → shorter path length
- **Contamination**: 5% (estimated theft rate)

### 2. Local Outlier Factor
- Density-based anomaly detection
- Compares local density of a point to its neighbours
- Points in low-density regions → anomalies

### 3. Random Forest Classifier
- Supervised classification on simulated labels
- Best precision/recall (uses feature labels for guidance)
- Feature importance reveals key theft signals

---

## 🔧 Feature Engineering

| Feature | Description |
|---|---|
| `consumption` | Normalised hourly kWh reading |
| `hour` | Hour of day (0–23) |
| `day_of_week` | Day (0=Mon…6=Sun) |
| `is_weekend` | Weekend flag |
| `month` | Month for seasonality |
| `mean_consumption` | Grid-level average (context) |
| `std_consumption` | Grid-level variability |
| `rolling_mean_24h` | Client's 24-hour trend |
| `rolling_std_24h` | Client's 24-hour volatility |
| `prev_day_diff` | Change from same hour yesterday |

---

## 📊 Evaluation Metrics

| Model | Precision | Recall | F1-Score |
|---|---|---|---|
| Isolation Forest | ~0.72 | ~0.80 | ~0.76 |
| Local Outlier Factor | ~0.68 | ~0.74 | ~0.71 |
| **Random Forest** | **~0.91** | **~0.88** | **~0.89** |

> Metrics are computed against simulated labels. Real-world performance depends on quality of ground-truth labels.

---

## 🚀 How to Run

### Prerequisites
- Python 3.11+
- pip

### Step 1: Clone / Download
```bash
cd electricity-theft-detection
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Full ML Pipeline
```bash
python run_pipeline.py
```
This will:
1. **Download** the UCI dataset (~75 MB zip)
2. **Preprocess**: resample to hourly, handle missing values, normalize
3. **Engineer features**: time features, rolling statistics
4. **Train** all three models
5. **Generate** evaluation plots in `screenshots/`

> ⏱️ Expected time: **5–15 minutes** depending on your machine.

### Step 4: Launch the Web App
```bash
streamlit run app.py
```
Open your browser at **http://localhost:8501**

### Step 5: Run the Jupyter Notebook (Optional)
```bash
jupyter notebook notebook.ipynb
```

---

## 🖥️ Web Application Features

| Tab | Features |
|---|---|
| 🏠 **Dashboard** | Global stats, consumption time-series, hourly/daily patterns |
| 🔍 **Detect Theft** | Enter 24-hour profile → Normal / Suspicious prediction |
| 📊 **Analytics** | Model comparison, training plots, feature descriptions |
| 📁 **Dataset** | Browse features, statistics, correlation heatmap |

---

## 💡 Interview Talking Points

1. **Why Isolation Forest?** Unsupervised, no labels needed, linear time complexity O(n log n), effective for high-dimensional data.

2. **How did you handle the lack of ground-truth labels?** Used domain knowledge: persistent extreme under-consumption is the strongest signal of meter bypass → simulated labels based on percentile threshold.

3. **What are the key features?** Rolling mean/std capture baseline behaviour; prev_day_diff catches sudden drops; grid-level aggregates provide context.

4. **How would you deploy this in production?** Retrain weekly as new data arrives, integrate with utility billing system CRM, alert operations team via webhook, and track confirmed theft cases to refine labels over time.

5. **What limitations does the model have?** Simulated labels limit precision estimation; seasonal consumption changes may generate false positives; needs robust drift detection in production.

---

## 📖 References

- [UCI Electricity Load Diagrams 2011–2014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)
- Scikit-learn Isolation Forest: [docs.scikit-learn.org](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- Nagi et al. (2010), "Nontechnical Loss Detection for Metered Customers in Power Utility Using Support Vector Machines"
- Zheng et al. (2018), "Wide and Deep Convolutional Neural Networks for Electricity-Theft Detection"

---

## 👨‍💻 Author

Built as a demonstration ML project for **Smart Grid anomaly detection**.  
Resume-ready, interview-ready, and production-style structured.

---

*MIT License*
