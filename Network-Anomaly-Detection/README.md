# Network Anomaly Detection

Network intrusion detection using **Random Forest** with multi-class traffic classification.

Built as part of the [Hack The Box Academy](https://academy.hackthebox.com/) **AI Red Teamer** path.

## Overview

This project builds a network intrusion detector using Random Forest. Trained on the [NSL-KDD dataset](https://www.unb.ca/cic/datasets/nsl.html) (~125,000 records), the model classifies network traffic as normal or one of four attack categories (DoS, Probe, Privilege Escalation, Access) with ~99.5% accuracy.

## Pipeline

```
Raw Traffic Record -> Encode Categorical Features -> Select Numeric Features -> Combine Features -> Map Attack Labels -> Train/Val/Test Split -> Train Random Forest -> Evaluate
```

## Tech Stack

- **Python**: pandas, NumPy, scikit-learn, seaborn, matplotlib
- **Model**: Random Forest Classifier (`n_estimators=100`, `random_state=1337`)
- **Features**: One-hot encoded categoricals (`protocol_type`, `service`) + 34 numeric traffic features
- **Task**: Multi-class classification (Normal, DoS, Probe, Privilege Escalation, Access)

## Results

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Normal | 0.99 | 1.00 | 1.00 | 15,402 |
| DoS | 1.00 | 1.00 | 1.00 | 10,721 |
| Probe | 0.99 | 1.00 | 1.00 | 2,796 |
| Privilege | 0.62 | 0.24 | 0.34 | 21 |
| Access | 0.96 | 0.92 | 0.94 | 764 |
| **Overall** | **0.99** | **0.99** | **0.99** | **29,704** |

> Privilege Escalation scores lower due to extreme class imbalance. There were only 21 test samples out of ~30k, so the model just doesn't see it enough during training.

## Quick Start

```bash
pip install numpy pandas scikit-learn seaborn matplotlib
```

Open `Network_Anomaly_Detection.ipynb` and run all cells. The notebook downloads the dataset automatically.

## Learn More

For a detailed walkthrough of the theory (Random Forest, feature engineering, multi-class classification) and step-by-step code explanations, see the [WALKTHROUGH.md](./WALKTHROUGH.md).
