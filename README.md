# Spatiotemporal NO2 Prediction using A3T-GCN

**Course:** CMP712 — Machine Learning
**Student:** Halil Burak Pala | Hacettepe University, Department of Computer Engineering

---

## Overview

This repository contains the implementation of the paper:

> Iskandaryan, D., Ramos, F., & Trilles, S. (2023). *Graph Neural Network for Air Quality Prediction: A Case Study in Madrid.* IEEE Access, 11, 2729–2742. https://doi.org/10.1109/ACCESS.2023.3234214

The goal is to replicate the Attention Temporal Graph Convolutional Network (A3T-GCN) approach for predicting hourly NO2 concentrations across 24 air quality monitoring stations in Madrid, Spain.

---

## Dataset

The dataset is **not included in this repository**. It can be accessed via Google Drive:

📁 **[Download Dataset](https://drive.google.com/drive/folders/1sxp8qMMNhS4J16W6eHHif3rXHj9yTjU8?usp=sharing)**

| File | Description |
|------|-------------|
| `Mad_Station_2019.csv` | Training set — Jan–Jun 2019, 24 stations × 4,344 hourly rows |
| `Mad_Station_2022.csv` | Test set — Jan–Jun 2022, 24 stations × 4,344 hourly rows |
| `distanceNodes.txt` | Pairwise Euclidean distances between stations (ArcPy output) |

Each row contains 18 features: NO2 (target), 5 meteorological variables, 4 traffic variables, and 8 one-hot encoded wind direction categories.

---

## Notebooks

### Progress Report 1 — Data Analysis & Graph Construction

<a href="https://colab.research.google.com/github/palahb/no2-prediction/blob/main/progress_report_1.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Covers: environment setup, EDA, graph construction from distance data, Z-score normalisation, and sliding window preprocessing.

### Progress Report 2 — Model Training and Evaluation

<a href="https://colab.research.google.com/github/palahb/no2-prediction/blob/main/progress_report_2.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Covers: training and evaluating four deep learning
models on the Madrid air quality dataset.

### Final Report 2 — Hyperparameter Sweep and Ablation Study

<a href="https://colab.research.google.com/github/palahb/no2-prediction/blob/main/final_report.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Covers: Fixing A3T-GCN training by addressing overfitting, hyperparameter sweep to find the best setup of A3T-GCN, ablation study to justify distance-based edge weights and per-station error analysis.

---

> Tested on Google Colab with Python 3.10, PyTorch 2.10.0, CUDA 12.8 (Tesla T4).