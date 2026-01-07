# SOC Log Anomaly Detection using Deep Learning

## 1. Introduction

Security Operation Centers (SOC) continuously monitor system logs to detect suspicious activities such as brute-force attacks, unauthorized access, or privilege abuse. However, the large volume and unstructured nature of logs make manual analysis difficult.

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

The system follows a complete pipeline from raw log ingestion to automated alert generation:

```
Raw Logs → Parse → Sequence → Model → Score → Alert
```

### Pipeline Phases

1. **Log Ingestion**: Collect and preprocess raw log data from various sources (e.g., auth.log, system.log)
2. **Parse**: Log template extraction using Drain3 parser to normalize and reduce data size
3. **Sequence**: Temporal sequence construction with sliding windows for model training
4. **Model**: Deep Learning model (LSTM-based DeepLog) training and inference
5. **Score**: Anomaly score computation using negative log-likelihood
6. **Alert**: Automated alert generation with severity classification (NONE, LOW, MED, HIGH)

---

## 4. Project Structure

```
soc-log-anomaly/
├── data/
│   ├── raw/              # Raw log files
│   ├── parsed/           # Parsed log data (templates)
│   ├── sequences/        # Sequence data and scores
│   └── results/          # Analysis results and visualizations
│
├── ingestion/
│   └── load_data.py      # Log data loading utilities
│
├── parsing/
│   ├── drain_parser.py   # Drain3 log parser
│   └── drain3.ini        # Drain3 configuration
│
├── sequence/
│   └── build_sequences.py  # Sequence construction with sliding windows
│
├── model/
│   ├── deeplog_lstm.py      # DeepLog LSTM architecture
│   ├── train.py             # Training script with validation & metrics
│   ├── infer.py             # Inference script for anomaly scoring
│   └── model_analysis.py    # Model analysis tools (architecture, embeddings)
│
├── scoring/
│   └── anomaly_score.py     # Anomaly score computation and threshold management
│
├── baseline/
│   ├── isolation_forest.py  # Isolation Forest baseline model
│   └── run_isolation_forest.py
│
├── evaluation/
│   ├── score_stats.py       # Score statistics
│   ├── compare_if_lstm.py   # Model comparison
│   ├── case_study.py        # Case study analysis
│   └── dl_metrics.py        # Deep Learning specific metrics
│
├── demo/
│   └── app.py               # Interactive demo application
│
├── notebook/
│   └── analysis.ipynb       # Comprehensive pipeline analysis notebook
│
├── utils.py
├── requirements.txt
└── README.md
```

---

## 5. Deep Learning Architecture

### DeepLog LSTM Model

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

- **Embedding Layer**: Learns dense representations of log templates (default: 16 dimensions)
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
4. **Alert Severity Classification**:
   - NONE: score < 95th percentile
   - LOW: 95th percentile ≤ score < 99th percentile
   - MED: 99th percentile ≤ score < 99.9th percentile
   - HIGH: score ≥ 99.9th percentile

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

- **Source**: Ubuntu `auth.log`
- **Format**: Plain text log files
- **Size**: Variable (typically 50,000 – 200,000 log lines)
- **Anomaly ratio**: Approximately 1–5%
- **Log Types**: Authentication logs, system logs

---

## 7. Evaluation Metrics

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
  - **Alert Statistics**: Severity distribution (NONE, LOW, MED, HIGH)

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
- Alert severity distribution

The LSTM model is comprehensively compared against the Isolation Forest baseline to demonstrate Deep Learning advantages.

---

## 8. Analysis Notebook

The `notebook/analysis.ipynb` provides comprehensive pipeline analysis following the complete workflow:

### Contents:

1. **Phase 1: Log Ingestion**
   - Raw log file analysis
   - Data source overview
   - Log entry statistics

2. **Phase 2: Parse**
   - Parsed data statistics
   - Template frequency analysis
   - Template distribution visualizations

3. **Phase 3: Sequence**
   - Sequence construction analysis
   - Sequence length statistics
   - Event frequency in sequences

4. **Phase 4: Model**
   - Model architecture analysis
   - Detailed model summary
   - Parameter counting
   - Training performance visualization
   - Training/validation loss curves
   - Accuracy progression

5. **Phase 5: Score**
   - Anomaly score statistics
   - Score distribution analysis
   - Perplexity calculation
   - Percentile-based threshold analysis
   - Cumulative distribution functions

6. **Phase 6: Alert**
   - Alert generation and analysis
   - Severity classification (NONE, LOW, MED, HIGH)
   - Alert statistics and visualizations
   - Model comparison (LSTM vs Isolation Forest)
   - Case studies of high anomaly sequences

7. **Conclusions and Future Directions**
   - Pipeline performance summary
   - Key findings and observations
   - Strengths and areas for improvement
   - Future work directions

---

## 9. Installation

### Requirements

Install required packages:

```bash
pip install -r requirements.txt
```

### Dependencies

Key dependencies include:
- Python 3.7+
- PyTorch
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn
- Drain3

---

## 10. Usage Examples

### Complete Pipeline Execution

#### Step 1: Log Ingestion

```bash
cd ingestion
python load_data.py
```

#### Step 2: Log Parsing

```bash
cd parsing
python drain_parser.py
```

#### Step 3: Sequence Construction

```bash
cd sequence
python build_sequences.py \
    --input_file ../data/parsed/parsed_data.json \
    --output_dir ../data/sequences/ \
    --window_size 20
```

#### Step 4: Model Training

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
    --val_split 0.2 \
    --output_dir .
```

#### Step 5: Anomaly Scoring

```bash
cd model
python infer.py \
    --sequences ../data/sequences/sequences.pkl \
    --ckpt model.pth \
    --out ../data/sequences/lstm_scores.pkl \
    --score_type nll
```

#### Step 6: Alert Generation

Alerts are automatically generated in the analysis notebook based on computed scores and thresholds.

### Baseline Model (Isolation Forest)

```bash
cd baseline
python run_isolation_forest.py
```

### Interactive Demo

```bash
cd demo
python app.py
```

### Model Analysis

```bash
python model/model_analysis.py \
    --ckpt model/model.pth \
    --num_labels <num_templates> \
    --output_dir data/results/
```

### Evaluation and Comparison

```bash
python evaluation/dl_metrics.py \
    --lstm_scores data/sequences/lstm_scores.pkl \
    --if_scores data/sequences/if_scores.pkl \
    --output_dir data/results/
```

---

## 11. Key Features

### Deep Learning Contributions

This project demonstrates several important Deep Learning concepts:

1. **Sequence Modeling**: LSTM captures temporal patterns in log sequences
2. **Unsupervised Learning**: Learns normal behavior without labeled anomalies
3. **Embedding Learning**: Automatically learns meaningful log template representations
4. **Transfer Learning Potential**: Embeddings can be reused across different log types
5. **Interpretability**: Model analysis tools provide insights into learned patterns

### System Features

- Complete end-to-end pipeline from logs to alerts
- Automated alert severity classification
- Comprehensive visualization and analysis tools
- Baseline comparison for validation
- Interactive demo application
- Modular architecture for easy extension

---

## 12. Results and Performance

### Model Performance

- The LSTM model achieves high accuracy in next-event prediction
- Effective anomaly detection with percentile-based thresholds
- Complementary detection signals compared to traditional ML methods

### Alert Generation

- Automated severity classification (NONE, LOW, MED, HIGH)
- Configurable thresholds based on percentiles
- Statistical analysis of alert distribution

### Comparison with Baseline

- LSTM and Isolation Forest capture different anomaly signals
- Deep Learning provides complementary detection capabilities
- Combined approach can improve overall detection accuracy

---

## 13. Future Work

### Deep Learning Enhancements

- **Transformer-based models**: BERT/Transformer for log sequences
- **Attention mechanisms**: Interpretable anomaly detection
- **Multi-task learning**: Joint prediction and classification
- **Transfer learning**: Pre-trained models for log analysis
- **Ensemble methods**: Combining multiple Deep Learning models

### System Improvements

- Support additional log types (network logs, application logs)
- Real-time log streaming with online learning
- Improved anomaly explanation with attention visualization
- Model compression for edge deployment
- Interactive dashboard for SOC analysts
- Alert prioritization based on context and historical patterns

### TO-DO works

- Multi-log source support
- Interactive web dashboard
- Alert prioritization logic
- Model explainability improvements (SHAP/LIME)

## 14. Team Work Distribution

Each team member is responsible for a specific pipeline stage:

- Raw log collection and ingestion
- Log parsing with Drain3
- Sequence construction
- Model development and training
- System integration
- Baseline modeling
- Evaluation and analysis

---

## 15. Conclusion

This project demonstrates that **sequence-based Deep Learning** can effectively detect anomalous behaviors in authentication logs, while remaining practical for SOC environments. The comprehensive analysis tools and visualizations make it easy to understand model behavior and validate Deep Learning advantages over traditional ML approaches.

### Deep Learning Advantages Demonstrated

- Better sequence pattern capture
- Automatic feature learning (embeddings)
- Scalable to large log volumes
- Complementary detection to traditional ML
- Rich analysis and visualization capabilities

The system follows a complete pipeline from log ingestion to automated alert generation, making it production-ready for SOC environments with proper configuration and tuning.

---

## 16. Contact

Vu.DNH235629@sis.hust.edu.vn

---

**Last Updated**: 08/01/2026
