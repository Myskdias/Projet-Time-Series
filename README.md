# AnomalyBERT for Multivariate Time Series Anomaly Detection

This repository contains our MVA 2025/2026 **Machine Learning for Time Series** mini-project on self-supervised anomaly detection for multivariate time series.

**Project report:** [read the report in the repository](./Projet_TS.pdf) · [download the PDF](https://raw.githubusercontent.com/Myskdias/Projet-Time-Series/main/Projet_TS.pdf)

The project is based on **AnomalyBERT**, a self-supervised Transformer architecture for time series anomaly detection using a data degradation scheme. We apply and adapt the method to the **Pooled Server Metrics (PSM)** dataset, then compare it against classical anomaly detection methods and improve the synthetic degradation process used during training.

## Project overview

Anomaly detection is central for monitoring critical systems such as servers, industrial processes, and network infrastructure. In many real-world settings, labeled anomalies are scarce or expensive to obtain. This motivates self-supervised methods that can learn from healthy data only.

AnomalyBERT follows this idea. During training, it starts from normal time series windows, injects synthetic anomalies into selected intervals, and trains a Transformer-based model to recover the artificial anomaly mask. At test time, the model is used to detect real anomalies.

This project studies whether this self-supervised strategy generalizes well to the PSM dataset and whether the degradation scheme can be improved when adapted to the dynamics of server metrics.

## Main contributions

This project makes four main contributions:

1. **Application to PSM**  
   We extend the original AnomalyBERT experiments by applying the method to the Pooled Server Metrics dataset.

2. **Classical baseline comparison**  
   We compare AnomalyBERT with several anomaly detection methods studied in the course:
   - adaptive Median / Median Absolute Deviation (MAD);
   - Vector Autoregression (VAR);
   - Matrix Profile.

3. **Optimization of the degradation scheme**  
   We empirically tune the probabilities of the synthetic perturbations used during training in order to better match the PSM dataset.

4. **Dataset-specific perturbations**  
   We introduce two new synthetic noise types tailored to server metrics:
   - **Uniform Shift**, which simulates sudden state changes;
   - **Linear Drift**, which simulates gradual degradation such as memory leaks.

## Dataset: Pooled Server Metrics (PSM)

The PSM dataset contains multivariate server telemetry collected from eBay production nodes. It includes 26 dimensions such as CPU usage, memory usage, and network traffic, over approximately 21 weeks.

In this project we use a preprocessed version of PSM with:

- **132,481** training timestamps;
- **87,841** test timestamps;
- **26** variables;
- a test anomaly rate of approximately **27.76%**.

The training set is considered anomaly-free, which makes it suitable for self-supervised training on normal data.

The preprocessing includes:

- linear interpolation for missing values;
- normalization across dimensions.

The data analysis shows several important properties of PSM:

- strong inter-variable dependencies;
- low-frequency seasonal structure;
- non-stationary local means;
- anomalies that may appear as broken correlations or frozen dynamics rather than simple spikes.

These properties explain why purely statistical methods can struggle on this dataset.

## Method: AnomalyBERT

AnomalyBERT operates on multivariate time series windows:

```text
X = x_{t0:t1} in R^{N x D}
```

where `N` is the window size and `D` is the number of variables.

The training pipeline is self-supervised:

1. sample a normal time series window;
2. select an interval inside the window;
3. inject a synthetic perturbation into that interval;
4. generate the corresponding artificial anomaly mask;
5. feed the degraded window into a Transformer;
6. train the model to predict the anomaly mask.

The model is optimized with a binary cross-entropy loss between the predicted anomaly scores and the artificial labels produced by the degradation step.

## Original degradation scheme

The original AnomalyBERT degradation scheme contains four perturbation types:

### Soft Replacement

The selected interval is replaced by a weighted combination of the original signal and an external segment. This creates subtle contextual anomalies while preserving local statistics.

### Uniform Replacement

The interval is replaced by a constant value. This can mimic signal saturation or frozen values.

### Peak Noise

A sharp point anomaly is injected into the signal. This corresponds to abrupt spikes.

### Length Adjustment

The interval is stretched or compressed in time. This simulates changes in local frequency or speed.

The probability of each perturbation can be adjusted, which allows the training distribution to be adapted to a specific dataset.

## Architecture

The model uses a Transformer-based architecture adapted to time series.

First, local time series patches are projected into a latent feature space by a linear embedding layer. These embedded features are then processed by a Transformer body.

Unlike standard Transformers that rely only on absolute positional encodings, AnomalyBERT introduces an **ID relative position bias** directly into the attention mechanism:

```text
Attention(Q, K, V) = Softmax(QK^T / sqrt(d) + B) V
```

where `B` is a learnable relative position bias matrix. This helps the model capture local and global temporal dependencies.

## Baselines

To evaluate whether a Transformer-based model is justified, we implemented three classical anomaly detection baselines from scratch using NumPy and PyTorch.

### Median / MAD

A robust statistical baseline based on deviations from the local median and median absolute deviation. It is useful for abrupt deviations but less effective when anomalies correspond to excessive regularity or broken correlations.

### VAR

Vector Autoregression models linear dependencies between time series variables. It can capture some inter-sensor relationships but remains limited for nonlinear server dynamics.

### Matrix Profile

Matrix Profile detects anomalous subsequences as discords, i.e. subsequences without close neighbors. It performs poorly on PSM because several real anomalies recur multiple times and therefore do not appear as isolated discords.

## Results

### Comparison with classical methods

| Method | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| AnomalyBERT | 0.48372 | 0.50746 | 0.49513 |
| MAD | 0.3815 | 0.3848 | 0.3832 |
| VAR | 0.3557 | 0.3588 | 0.3572 |
| Matrix Profile | 0.1936 | 0.1953 | 0.1945 |

AnomalyBERT clearly outperforms the classical baselines. This supports the relevance of learning temporal context and nonlinear dependencies for this dataset.

### Degradation scheme sensitivity

We then studied the impact of each original synthetic perturbation by increasing its probability while keeping the others at a baseline value.

| Dominant perturbation | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| Soft Replacement | 0.45738 | 0.49436 | 0.47515 |
| Uniform Replacement | 0.41849 | 0.45232 | 0.43475 |
| Peak Noise | 0.40972 | 0.44284 | 0.42564 |
| Length Adjustment | 0.31088 | 0.33784 | 0.32574 |

Soft Replacement is the most useful perturbation on PSM. Length Adjustment performs poorly because server metrics follow a fixed sampling frequency, making time-stretching artifacts unrealistic for this dataset.

The optimized weighting scheme improves the F1-score from **0.49513** to **0.50561**.

### Custom perturbations for PSM

We added two new degradation types to better match realistic server failures:

- **Uniform Shift:** adds a constant offset to a segment, simulating a sudden state change such as a service crash or restart;
- **Linear Drift:** adds a progressive ramp, simulating gradual degradation such as a memory leak.

With these domain-specific perturbations, the model reaches better generalization performance, with F1-scores between **0.51204** and **0.53987** across several runs.

## Training behavior

The training curves show a clear gap between the self-supervised proxy task and real anomaly detection.

The model learns to detect synthetic anomalies very quickly: training accuracy reaches nearly 100%, and the loss becomes close to zero after a small number of iterations. However, the test metrics remain volatile. This indicates that the synthetic task can become too easy and that the model may overfit to artificial perturbation patterns.

This observation motivates the central improvement of the project: designing more realistic perturbations that force the model to learn meaningful temporal and inter-variable structure instead of memorizing obvious artifacts.

## Code contributions

This repository is based on the original AnomalyBERT codebase:

```text
https://github.com/Jhryu30/AnomalyBERT
```

The main project-specific additions are:

- modification of `train.py` to add two new perturbation types;
- addition of `baseline.ipynb`, containing the classical baseline implementations and evaluation metrics;
- addition of `my_test.ipynb`, used to test and evaluate predictions from trained models;
- implementation of custom Recall, Precision, and F1-score evaluation logic based on the original metric behavior;
- empirical tuning of the degradation probabilities for PSM.

## Installation

The project follows the original AnomalyBERT setup. A typical installation is:

```bash
git clone https://github.com/Myskdias/Projet-Time-Series.git
cd Projet-Time-Series

conda create --name anomalybert python=3.8
conda activate anomalybert

pip install -r requirements.txt
```

Depending on your CUDA version, you may need to install the appropriate PyTorch build manually before installing the rest of the requirements.

## Data preparation

Prepare the dataset following the instructions in:

```text
utils/DATA_PREPARATION.md
```

Then edit the dataset path in:

```text
utils/config.py
```

For example:

```python
DATASET_DIR = "path/to/dataset/processed/"
```

## Training examples

Train AnomalyBERT on PSM with the default configuration:

```bash
python3 train.py --dataset=PSM
```

Train with a custom degradation scheme:

```bash
python3 train.py --dataset=PSM \
  --soft_replacing=0.5 \
  --uniform_replacing=0.1 \
  --peak_noising=0.1 \
  --length_adjusting=0.1
```

The exact options depend on the current version of `train.py` and the additional perturbations integrated for this project.

## Repository contents

```text
.
├── train.py                 # Training code, including added perturbations
├── baseline.ipynb           # Classical baselines: MAD, VAR, Matrix Profile
├── my_test.ipynb            # Model evaluation notebook
├── utils/                   # Configuration and data preparation utilities
├── requirements.txt         # Python dependencies
├── Projet_TS.pdf            # Full project report
└── README.md                # Project overview
```

## Reference

This project is based on:

```text
Y. Jeong, E. Yang, J. H. Ryu, I. Park, and M. Kang.
AnomalyBERT: Self-supervised Transformer for Time Series Anomaly Detection using Data Degradation Scheme.
2023.
```

Additional references include the Matrix Profile paper and the PSM dataset paper.

## Conclusion

This project shows that self-supervised Transformer-based anomaly detection can outperform classical statistical methods on PSM, but also that the quality of the synthetic degradation scheme is crucial.

The main lesson is that synthetic anomalies must be close enough to real failure modes. Generic perturbations such as peaks or time stretching are not always appropriate for server metrics. Dataset-specific perturbations such as uniform shifts and linear drifts provide a better proxy task and lead to improved anomaly detection performance.
