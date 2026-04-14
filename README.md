# Anomaly Detection in Time Series

A study of anomaly detection techniques applied to real-world time series data, covering the full methodological spectrum from classical statistical baselines to modern deep learning architectures across two independent datasets.

---

## Datasets

### EC2 CPU Utilization — Numenta Anomaly Benchmark (NAB)

A real univariate CPU monitoring signal from an Amazon Web Services EC2 instance, part of the well-established NAB benchmark suite.

- **Source:** [Numenta Anomaly Benchmark (NAB) — Kaggle](https://www.kaggle.com/datasets/boltzmannbrain/nab)
- **Signal:** AWS EC2 CPU utilization percentage
- **Sampling rate:** One reading every 5 minutes
- **Total observations:** 4,032 sequential data points

### Cloud Resource Usage — Multivariate Infrastructure Monitoring

A multivariate cloud dataset recording four hardware metrics simultaneously across a multi-tenant server environment including simulated cryptomining attack events.

- **Source:** [Cloud Resource Usage Dataset for Anomaly Detection — Kaggle](https://www.kaggle.com/datasets/programmer3/cloud-resource-usage-dataset-for-anomaly-detection)
- **Sampling rate:** One reading per minute
- **Duration:** 10 continuous days
- **Total records:** 14,400 rows (`60 min × 24 h × 10 days`)
- **Features:** `CPU_Usage`, `Memory_Usage`, `Disk_IO`, `Network_IO`
- **Workloads:** Web Service, Video Streaming, Database Query, and simulated Cryptomining anomalies

---

## Track 1 — EC2 CPU Utilization: Statistical and Machine Learning

Applied to the NAB EC2 dataset, these methods evaluate each time point independently and serve as calibrated baselines against the deep learning approaches.

### Approaches

**1. Median Absolute Deviation (MAD)**

A robust univariate statistical baseline that resists distortion from the very outliers it detects.

- Computes a robust Z-Score for every observation: `Z = 0.6745 * (x - median) / MAD`
- Any point exceeding an absolute robust Z-Score of 3.5 is classified as anomalous
- Computationally inexpensive and parameter-light

**2. Isolation Forest**

An unsupervised ensemble method designed explicitly for anomaly detection.

- Builds multiple random decision trees; anomalous points are isolated closer to the root with shorter average path lengths
- Requires a `contamination` parameter estimating the expected anomaly proportion
- Scales well to higher-dimensional data

**3. Local Outlier Factor (LOF)**

A density-based algorithm that finds locally irregular points rather than global extremes.

- Computes the local density deviation of each point relative to its `k` nearest neighbors
- Effective at detecting anomalies that are unremarkable globally but unusual within their neighborhood

---

## Track 2 — Cloud Resource Usage: Deep Learning Autoencoders

Applied to the multivariate cloud dataset. Instead of evaluating points independently, these models learn the temporal structure of normal multi-metric behavior and flag deviations through reconstruction error.

### Dataset Visualization

The figure below shows all four cloud metrics over the full recording period. Red vertical lines mark known anomaly events (cryptomining spikes) across the features simultaneously.

![Cloud Dataset Signal with Anomaly Labels](assets/DATASET_rawsignal%20with%20redlines%20as%20anomalies%20across%20diffrent%20features.png)

### Architecture

Both models share an identical **Encoder–Decoder** topology:

- **Input:** Sliding window tensors of shape `(samples, 60 timesteps, 4 features)` — one hour of history per prediction
- **Encoder:** Compresses the 60-step multivariate sequence into a latent bottleneck representation
- **Decoder:** Reconstructs the original sequence from the latent space
- **Anomaly Threshold:** `T = μ + 3σ` of the training MAE distribution — any test window exceeding this is flagged
- **Training:** Only on anomaly-free windows (dynamically filtered) to ensure a pure normal baseline
- **Regularization:** Dropout layers and Early Stopping (patience = 5, min delta = 0.001)

---

### 4. LSTM Autoencoder

Uses **Long Short-Term Memory** cells, effective at capturing long-range temporal dependencies.

**Training convergence — Training MAE vs Validation MAE:**

![LSTM Training vs Validation Loss](assets/LSTM_MAE_TraininvsValidationgraph(trainblue_validationyellow).png)

**Detection output — Test sequence reconstruction with flagged anomalies:**

![LSTM Predictions Overlaid on Ground Truth](assets/LSTM_GRAPH_Test%20data%20overlayed%20with%20predictions%20vs%20groud%20Truth.png)

| Metric    | Score  |
|-----------|--------|
| Precision | 99.46% |
| Recall    | 81.89% |
| F1-Score  | 89.82% |

---

### 5. GRU Autoencoder

Uses **Gated Recurrent Unit** cells, which merge the forget and input gates into a single update gate, reducing parameter count and training time while achieving superior recall.

**Training convergence — Training MAE vs Validation MAE:**

![GRU Training vs Validation Loss](assets/GRU_MAE_TraininvsValidationgraph(trainblue_validationyellow)).png)

**Detection output — Test sequence reconstruction with flagged anomalies:**

![GRU Predictions Overlaid on Ground Truth](assets/GRU_GRAPH_Test%20data%20overlayed%20with%20predictions%20vs%20groud%20Truth.png)

| Metric    | Score  |
|-----------|--------|
| Precision | 99.48% |
| Recall    | 85.10% |
| F1-Score  | **91.73%** |

---

### Comparative Results

| Model            | Precision | Recall     | F1-Score      |
|------------------|-----------|------------|---------------|
| LSTM Autoencoder | 99.46%    | 81.89%     | 89.82%        |
| GRU Autoencoder  | 99.48%    | **85.10%** | **91.73%**    |

The GRU Autoencoder outperforms the LSTM on both Recall and F1-Score while being computationally lighter, making it the preferred architecture for real-time cloud monitoring. The improvement in Recall directly translates to fewer missed cryptomining incidents in production.

---

## Repository Structure

```
Deep-Learning-Anomaly-Detection-TimeSeries-LSTM/
|
|-- data/
|   |-- ec2_cpu_utilization.csv          # EC2 CPU time series (NAB)
|   `-- cloud_dataset.csv                # Multivariate cloud resource dataset
|
|-- notebooks/
|   |-- Anomaly_Detection_TimeSeries_LSTM.ipynb  # Track 1: MAD, Isolation Forest, LOF
|   |-- Cloud_LSTM_Anomaly_Detection.ipynb       # Track 2: LSTM Autoencoder
|   `-- Cloud_GRU_Anomaly_Detection.ipynb        # Track 2: GRU Autoencoder
|
|-- assets/                              # Plots and visualizations used in this README
|
`-- README.md
```

---

## Setup

```bash
pip install pandas numpy matplotlib scikit-learn scipy tensorflow jupyter
```

**Run Track 1 (Statistical & ML baselines):**
```bash
jupyter notebook notebooks/Anomaly_Detection_TimeSeries_LSTM.ipynb
```

**Run Track 2 (Deep Learning autoencoders):**
```bash
jupyter notebook notebooks/Cloud_LSTM_Anomaly_Detection.ipynb
jupyter notebook notebooks/Cloud_GRU_Anomaly_Detection.ipynb
```

All notebooks are fully documented with markdown explanations at each step.

---

## Authors
Khalil AIT NOUISSE — Badr LAKLACH

Supervised by **Pr. Fatima Zohra El Hlouli** — ENSAM, Filiere Genie Informatique et Systemes Intelligents
