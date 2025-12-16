# 🛡️ SOC Log Anomaly Detection using Deep Learning

## 1. Introduction

Security Operation Centers (SOC) continuously monitor system logs to detect suspicious activities such as brute-force attacks, unauthorized access, or privilege abuse.  
However, the large volume and unstructured nature of logs make manual analysis difficult.

This project focuses on **log-based anomaly detection** using **Deep Learning**, specifically a **sequence-based LSTM model (DeepLog-style)**, to automatically learn normal system behavior and detect anomalous log sequences.

---

## 2. Project Scope

- **Log type**: Ubuntu `auth.log`
- **Operating system**: Linux (Ubuntu)
- **Learning paradigm**: Unsupervised / Semi-supervised
- **Main task**: Sequence anomaly detection
- **Target users**: SOC analysts

To ensure feasibility within limited time, the project focuses on **a single log type and a single deep learning model**, compared against a classical machine learning baseline.

---

## 3. System Pipeline

```
Raw Logs → Parsing → Sequence Construction → LSTM Model → Anomaly Scoring → Evaluation / Demo
```

Pipeline steps:
1. Log ingestion from `auth.log`
2. Log parsing using Drain
3. Sequence construction using sliding windows
4. LSTM-based anomaly detection (DeepLog)
5. Alert scoring and evaluation

---

## 4. Project Structure

```
soc-log-anomaly/
├── data/
│   ├── raw/
│   ├── parsed/
│   └── sequences/
│
├── ingestion/
│   └── load_logs.py
│
├── parsing/
│   └── drain_parser.py
│
├── sequence/
│   └── build_sequences.py
│
├── baseline/
│   └── isolation_forest.py
│
├── model/
│   ├── deeplog_lstm.py
│   ├── train.py
│   └── infer.py
│
├── scoring/
│   └── anomaly_score.py
│
├── evaluation/
│   └── metrics.py
│
├── demo/
│   └── app.py
│
├── notebooks/
│   └── analysis.ipynb
│
├── utils.py
├── requirements.txt
└── README.md
```

---

## 5. Models

### Baseline
- **Isolation Forest**
- Used for comparison with deep learning results

### Deep Learning Model
- **LSTM (DeepLog-style)**
- Predicts next log event in a sequence
- An anomaly is detected if the true event is not within top-k predictions

---

## 6. Dataset

- Source: Ubuntu `auth.log`
- Format: Plain text
- Size: ~50,000 – 200,000 log lines
- Anomaly ratio: ~1–5%

---

## 7. Evaluation Metrics

- Precision
- Recall
- F1-score

The LSTM model is compared against the Isolation Forest baseline.

---

## 8. Notebook

A single notebook `analysis.ipynb` is used for:
- Log exploration
- Parsing validation
- Sequence analysis
- Result visualization
- Discussion of limitations

---

## 9. Team Work Distribution

Each team member is responsible for a specific pipeline stage:
- Raw log collection
- Log parsing
- Sequence construction
- System integration, baseline modeling, and evaluation

---

## 10. Future Work

- Support additional log types
- Apply transformer-based models
- Integrate real-time log streaming
- Improve anomaly explanation

---

## 11. Conclusion

This project demonstrates that **sequence-based deep learning** can effectively detect anomalous behaviors in authentication logs, while remaining practical for SOC environments.
