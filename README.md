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
│   ├── deeplog_lstm.py          # DeepLog LSTM architecture
│   ├── train.py                  # Training script with validation & metrics
│   ├── infer.py                  # Inference script for anomaly scoring
│   └── model_analysis.py         # Model analysis tools (architecture, embeddings)
│
├── scoring/
│   └── anomaly_score.py
│
├── evaluation/
│   ├── score_stats.py            # Score statistics
│   ├── compare_if_lstm.py       # Model comparison
│   ├── case_study.py            # Case study analysis
│   └── dl_metrics.py            # Deep Learning specific metrics
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

## 5. Deep Learning Architecture

### 🧠 DeepLog LSTM Model

This project implements a **sequence-based Deep Learning model** inspired by DeepLog for log anomaly detection.

#### Architecture Details

```
Input: Sequence of log template IDs [t₁, t₂, ..., tₙ]
  ↓
Embedding Layer (num_labels → embedding_dim)
  ↓
LSTM Layer(s) (hidden_size, num_layers)
  ↓
Fully Connected Layer (hidden_size → num_labels)
  ↓
Output: Logits for next log template prediction
```

**Key Components:**
- **Embedding Layer**: Learns dense representations of log templates (default: 16-dim)
- **LSTM Layer**: Captures temporal dependencies in log sequences (default: 128 hidden units, 1 layer)
- **Fully Connected Layer**: Maps LSTM output to log template predictions

#### Training Process

The model is trained using:
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam (learning rate: 1e-3)
- **Training Strategy**: 
  - Train/Validation split (default: 80/20)
  - Sliding window approach for sequence generation
  - Batch training with configurable batch size
- **Metrics Tracked**:
  - Training & Validation Loss
  - Top-1 Accuracy
  - Top-k Accuracy (k=5)
  - Training curves visualization

#### Anomaly Detection Mechanism

1. **Next Event Prediction**: Model predicts the next log template in a sequence
2. **Anomaly Scoring**: 
   - **NLL (Negative Log-Likelihood)**: Lower probability → Higher anomaly score
   - **Top-k Method**: Anomaly if true event not in top-k predictions
3. **Threshold Selection**: Percentile-based (e.g., top 5% or 1% as anomalies)

#### Model Analysis Tools

The project includes comprehensive Deep Learning analysis tools:

- **Model Architecture Summary**: Parameter counting, layer breakdown, model size
- **Embedding Visualization**: t-SNE/PCA visualization of learned embeddings
- **Training Curves**: Loss and accuracy visualization
- **Gradient Flow Analysis**: Debugging training issues
- **Activation Distribution**: Understanding model internals

### Baseline Model

- **Isolation Forest**: Classical ML baseline for comparison
- Used to demonstrate Deep Learning advantages in sequence modeling

---

## 6. Dataset

- Source: Ubuntu `auth.log`
- Format: Plain text
- Size: ~50,000 – 200,000 log lines
- Anomaly ratio: ~1–5%

---

## 7. Deep Learning Evaluation Metrics

### Model Performance Metrics

- **Training Metrics**:
  - Loss (Cross-Entropy)
  - Top-1 Accuracy
  - Top-k Accuracy (k=5)
  - Training/Validation split performance

- **Anomaly Detection Metrics**:
  - **Perplexity**: Model uncertainty measure (exp(mean(NLL)))
  - **Score Distribution**: Statistical analysis of anomaly scores
  - **Percentile Thresholds**: Top 1%, 5%, 10% anomalies

- **Model Comparison**:
  - Spearman/Pearson correlation with baseline
  - Jaccard index for top-k overlap
  - Rank correlation analysis

### Visualization Tools

- Training curves (loss, accuracy over epochs)
- Embedding visualizations (t-SNE, PCA)
- Score distribution analysis
- Model comparison plots
- Case study visualizations

The LSTM model is comprehensively compared against the Isolation Forest baseline to demonstrate Deep Learning advantages.

---

## 8. Deep Learning Analysis Notebook

The `notebook/analysis.ipynb` provides comprehensive Deep Learning analysis:

### Contents:

1. **Model Architecture Analysis**
   - Detailed model summary
   - Parameter counting
   - Layer-wise breakdown
   - Model complexity analysis

2. **Training Performance Visualization**
   - Training/validation loss curves
   - Accuracy progression
   - Top-k accuracy trends
   - Learning progress metrics

3. **Embedding Analysis**
   - t-SNE/PCA visualization of learned embeddings
   - Embedding similarity matrix
   - Template clustering analysis

4. **Anomaly Score Analysis**
   - Score distribution statistics
   - Perplexity calculation
   - Percentile-based threshold analysis
   - Cumulative distribution functions

5. **Model Comparison (LSTM vs Isolation Forest)**
   - Score correlation analysis
   - Top-k overlap visualization
   - Rank comparison
   - Complementary detection insights

6. **Case Studies**
   - High-anomaly sequence analysis
   - Pattern identification
   - Real-world anomaly examples

---

## 9. Team Work Distribution

Each team member is responsible for a specific pipeline stage:
- Raw log collection
- Log parsing
- Sequence construction
- System integration, baseline modeling, and evaluation

---

## 10. Usage Examples

### Training the Deep Learning Model

```bash
cd model
python train.py \
    --sequences ../data/sequences/sequences.pkl \
    --window_size 5 \
    --embedding_dim 16 \
    --hidden_size 128 \
    --num_layers 1 \
    --batch_size 64 \
    --epochs 10 \
    --lr 1e-3 \
    --val_split 0.2
```

### Running Inference

```bash
python infer.py \
    --sequences ../data/sequences/sequences.pkl \
    --ckpt model.pth \
    --out ../data/sequences/lstm_scores.pkl \
    --score_type nll
```

### Model Analysis

```bash
python model_analysis.py \
    --ckpt model/model.pth \
    --num_labels <num_templates> \
    --output_dir results/
```

### Deep Learning Evaluation

```bash
python evaluation/dl_metrics.py \
    --lstm_scores data/sequences/lstm_scores.pkl \
    --if_scores data/sequences/if_scores.pkl \
    --output_dir results/
```

## 11. Future Work

### Deep Learning Enhancements
- **Transformer-based models**: BERT/Transformer for log sequences
- **Attention mechanisms**: Interpretable anomaly detection
- **Multi-task learning**: Joint prediction and classification
- **Transfer learning**: Pre-trained models for log analysis
- **Ensemble methods**: Combining multiple Deep Learning models

### System Improvements
- Support additional log types
- Real-time log streaming with online learning
- Improved anomaly explanation with attention visualization
- Model compression for edge deployment

---

## 12. Key Deep Learning Contributions

This project demonstrates several important Deep Learning concepts:

1. **Sequence Modeling**: LSTM captures temporal patterns in log sequences
2. **Unsupervised Learning**: Learns normal behavior without labeled anomalies
3. **Embedding Learning**: Automatically learns meaningful log template representations
4. **Transfer Learning Potential**: Embeddings can be reused across different log types
5. **Interpretability**: Model analysis tools provide insights into learned patterns

## 13. Conclusion

This project demonstrates that **sequence-based Deep Learning** can effectively detect anomalous behaviors in authentication logs, while remaining practical for SOC environments. The comprehensive analysis tools and visualizations make it easy to understand model behavior and validate Deep Learning advantages over traditional ML approaches.

### Deep Learning Advantages Demonstrated:
- ✅ Better sequence pattern capture
- ✅ Automatic feature learning (embeddings)
- ✅ Scalable to large log volumes
- ✅ Complementary detection to traditional ML
- ✅ Rich analysis and visualization capabilities
