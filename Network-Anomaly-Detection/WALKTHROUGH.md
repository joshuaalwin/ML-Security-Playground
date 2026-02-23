# Network Anomaly Detection — Walkthrough

A step-by-step guide covering the theory and implementation behind this network intrusion detection project. If you're here to learn, this is for you.

## Table of Contents

- [1. Theory: Anomaly Detection & Random Forest](#1-theory-anomaly-detection--random-forest)
- [2. Preparing the Dataset](#2-preparing-the-dataset)
- [3. Preprocessing the Dataset](#3-preprocessing-the-dataset)
- [4. Splitting into Train / Validation / Test](#4-splitting-into-train--validation--test)
- [5. Training and Evaluation](#5-training-and-evaluation)
- [6. Saving and Loading the Model](#6-saving-and-loading-the-model)

---

## 1. Theory: Anomaly Detection & Random Forest

### What is Anomaly Detection?

An anomaly is something that deviates significantly from the norm. In network security, anomalies can indicate:

- Malicious activities
- Network intrusions
- Security breaches

The goal is to train a model on labelled traffic — normal and attack — so it can automatically flag suspicious connections in the future.

### Random Forest

Random Forest is an ML algorithm that builds multiple decision trees and aggregates their predictions. Each tree votes for a class, and the class receiving the **majority of votes** is chosen.

> For regression problems, the final output is the **average** of the individual tree outputs rather than a majority vote.

Three key concepts shape the construction of a Random Forest:

1. **Bootstrapping** — Multiple subsets of the training data are created via sampling with replacement. Each subset trains a separate decision tree.
2. **Tree Construction** — For each tree, a random subset of features is considered at every split, ensuring diversity and reducing correlations among trees.
3. **Voting** — After all trees are trained, classification involves majority voting, while regression involves averaging predictions.

### A Single Decision Tree

Before the forest, understand one tree. A decision tree splits data based on feature thresholds:

```
Is src_bytes > 1000?
├── Yes → Is serror_rate > 0.5? → DoS
└── No  → Is rerror_rate > 0.3? → Probe
                               → Normal
```

The problem: a single tree can memorize training data (overfitting), making it brittle on unseen traffic. A Random Forest corrects this by averaging many diverse trees.

| Property | Single Tree | Random Forest |
|----------|-------------|---------------|
| Overfitting risk | High | Low |
| Variance | High | Low |
| Training speed | Fast | Slower |
| Interpretability | Easy | Harder |

### Why It Works for Network Anomaly Detection

Network traffic features (byte counts, error rates, connection flags) are heterogeneous — some categorical, some continuous, some near-zero for most samples. Random Forest handles this naturally **without requiring feature normalization or scaling**, making it well-suited to this problem.

---

## 2. Preparing the Dataset

### Downloading the Dataset

```python
import requests, zipfile, io

# URL for the NSL-KDD dataset
url = "https://academy.hackthebox.com/storage/modules/292/KDD_dataset.zip"

# Download the zip file and extract its contents
response = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(response.content))
z.extractall('.')  # Extracts to the current directory
```

### Importing Libraries

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
```

- `numpy` and `pandas` handle data loading and manipulation
- `RandomForestClassifier` provides the algorithm for anomaly detection
- `train_test_split` and `sklearn.metrics` support model evaluation and validation
- `seaborn` and `matplotlib` assist in visualizing distributions and model results

### Defining Column Names and Loading the Data

The NSL-KDD file has no header row, so column names are defined manually:

```python
file_path = r'KDD+.txt'

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'level'
]

df = pd.read_csv(file_path, names=columns)
print(df.head())
```

> These column names ensure each feature and label is properly identified. They include generic network statistics (`duration`, `src_bytes`, `dst_bytes`), categorical fields (`protocol_type`, `service`), and labels (`attack`, `level`) which classify the type of traffic observed.

---

## 3. Preprocessing the Dataset

The main goal is to transform raw network traffic data into a usable numeric format. We:

- Turn the text label into a **binary label** (Normal vs. Attack) or a **multi-class label** (Normal / DoS / Probe / Privilege / Access)
- Turn categorical columns like `protocol_type` and `service` into numbers using one-hot encoding
- Select which numeric columns to feed into the model
- Split the processed data into train / validation / test sets for evaluation

> **End result:** A clean `train_set` matrix of features (all numeric) and a target `multi_y` ready to plug into a classifier.

### 3.1 Creating a Binary Classification Target

This teaches the model a simple thing — is this connection normal or an attack?

```python
df['attack_flag'] = df['attack'].apply(lambda a: 0 if a == 'normal' else 1)
```

- `.apply` goes row by row on the `attack` column
- The lambda returns `0` (normal) or `1` (attack) for each value
- You get a clean numeric binary label that algorithms like Random Forest can use directly

### 3.2 Multi-Class Classification Target

Binary classification loses information. Knowing it's a DoS attack versus a Probe matters significantly for a defender — they require completely different responses. We instead distinguish between attack families:

```python
dos_attacks = ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod',
               'processtable', 'smurf', 'teardrop', 'udpstorm', 'worm']
probe_attacks = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
privilege_attacks = ['buffer_overflow', 'loadmdoule', 'perl', 'ps',
                     'rootkit', 'sqlattack', 'xterm']
access_attacks = ['ftp_write', 'guess_passwd', 'http_tunnel', 'imap',
                  'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack',
                  'snmpguess', 'spy', 'warezclient', 'warezmaster',
                  'xclock', 'xsnoop']

def map_attack(attack):
    if attack in dos_attacks:         return 1
    elif attack in probe_attacks:     return 2
    elif attack in privilege_attacks: return 3
    elif attack in access_attacks:    return 4
    else:                             return 0  # normal

df['attack_map'] = df['attack'].apply(map_attack)
```

Think of `map_attack` as a lookup table: string label → numeric class. `0` is normal; `1–4` are the four attack families.

| Class | Label | Description |
|-------|-------|-------------|
| Normal | 0 | Legitimate network traffic |
| DoS | 1 | Denial-of-Service — floods or exhausts target resources |
| Probe | 2 | Reconnaissance — scanning and probing for vulnerabilities |
| Privilege | 3 | Privilege escalation — gaining root or admin access |
| Access | 4 | Unauthorized access — exploiting credentials or services |

### 3.3 Encoding Categorical Variables

ML models require numeric inputs. Two columns are categorical strings — `protocol_type` (`tcp`, `udp`, `icmp`) and `service` (`http`, `ftp`, `smtp`, etc.) — which need to be encoded.

```python
features_to_encode = ['protocol_type', 'service']
encoded = pd.get_dummies(df[features_to_encode])
```

`pd.get_dummies` creates a column for each distinct category:
- `protocol_type_tcp`, `protocol_type_udp`, `protocol_type_icmp`
- Each new column is `1` if the row has that category, `0` otherwise

### 3.4 Selecting Numeric Features

NSL-KDD has many numeric features: some basic (duration, bytes), some content-based (`num_shells`), some traffic-based (`same_srv_rate`, etc.). We keep those that draw the strongest correlation to attack types:

```python
numeric_features = [
    'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'
]
```

Key features and what they signal:

| Feature | What it captures |
|---------|-----------------|
| `src_bytes` / `dst_bytes` | Data volume in each direction |
| `serror_rate` | Rate of SYN errors — high in SYN flood (DoS) |
| `rerror_rate` | Rate of REJ errors — high in port scans (Probe) |
| `root_shell` | Whether a root shell was obtained |
| `num_failed_logins` | Number of failed login attempts |
| `same_srv_rate` | Fraction of connections to the same service (high in DoS) |
| `diff_srv_rate` | Fraction to different services (high in scanning) |

### 3.5 Building the Final Feature Matrix

Combine the encoded categorical features and the numeric features into one matrix:

```python
train_set = encoded.join(df[numeric_features])
multi_y = df['attack_map']  # multi-class target vector
```

The final feature matrix is a horizontal concatenation of all one-hot columns and the 34 numeric columns — fully numeric and ready for training.

---

## 4. Splitting into Train / Validation / Test

We separate data into parts we train on, tune on, and evaluate on.

### Train vs. Test

```python
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(
    train_set, multi_y, test_size=0.2, random_state=1337
)
```

- `test_size=0.2` → 20% is held out as the **test set**
- 80% remains for training
- `random_state=1337` makes the split reproducible (same shuffle every run)

> The test set represents future unseen traffic — it is touched **only once** at the very end to estimate real-world detection performance.

### Training vs. Validation

The 80% training portion is further split into a training set and a validation set. This is where model tuning happens:

```python
multi_train_X, multi_val_X, multi_train_y, multi_val_y = train_test_split(
    train_X, train_y, test_size=0.3, random_state=1337
)
```

If the original dataset = 100%:

| Split | Size |
|-------|------|
| Test | 20% |
| Validation | 0.8 × 0.3 = **24%** |
| Train (for fitting) | 0.8 × 0.7 = **56%** |

Roles:

- `multi_train_X, multi_train_y` — used to fit the model
- `multi_val_X, multi_val_y` — used to tune hyperparameters and pick the best configuration
- `test_X, test_y` — touched only after tuning is done, for final performance numbers

> This structure prevents **data leakage**: hyperparameters are never chosen based on the test set, so the final test score reflects true generalization.

---

## 5. Training and Evaluation

### Training the Random Forest

```python
rf_model_multi = RandomForestClassifier(random_state=1337)
rf_model_multi.fit(multi_train_X, multi_train_y)
```

What's happening:

- `RandomForestClassifier` builds 100 decision trees on bootstrapped samples and averages their votes — this works extremely well for NSL-KDD multi-class intrusion detection
- `random_state=1337` makes the forest structure reproducible
- `.fit(multi_train_X, multi_train_y)` lets the forest learn how feature patterns map to: Normal, DoS, Probe, Privilege, Access

### Evaluation on the Validation Set

```python
multi_predictions = rf_model_multi.predict(multi_val_X)
```

`.predict` takes each validation connection and outputs one of the 5 class labels.

Computing metrics, confusion matrix, and classification report:

```python
accuracy  = accuracy_score(multi_val_y, multi_predictions)
precision = precision_score(multi_val_y, multi_predictions, average='weighted')
recall    = recall_score(multi_val_y, multi_predictions, average='weighted')
f1        = f1_score(multi_val_y, multi_predictions, average='weighted')

print(f"Validation Set Evaluation:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

conf_matrix = confusion_matrix(multi_val_y, multi_predictions)
class_labels = ['Normal', 'DoS', 'Probe', 'Privilege', 'Access']

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Network Anomaly Detection - Validation Set')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(classification_report(multi_val_y, multi_predictions, target_names=class_labels))
```

### Validation Results

```
Accuracy:  0.9950
Precision: 0.9949
Recall:    0.9950
F1-Score:  0.9949
```

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Normal | 0.99 | 1.00 | 1.00 | 18,519 |
| DoS | 1.00 | 1.00 | 1.00 | 12,784 |
| Probe | 0.99 | 0.99 | 0.99 | 3,409 |
| Privilege | 0.82 | 0.38 | 0.51 | 24 |
| Access | 0.97 | 0.92 | 0.95 | 908 |

![Validation Confusion Matrix](Validation-Confusion-Matrix.png)

What the numbers mean:

- **Accuracy 0.9950** — ~99.5% of all validation flows are classified into the correct class
- **Precision 0.9949 (weighted)** — when the model predicts a given class, it is correct ~99.5% of the time (averaged by class support)
- **Recall 0.9950 (weighted)** — the model successfully recovers ~99.5% of the actual samples in each class (averaged by support)
- **F1 0.9949 (weighted)** — the overall trade-off between precision and recall is extremely strong; typical for high-performing NSL-KDD classifiers

**Why Privilege Escalation underperforms:** Only 24 validation samples total. The model rarely encounters this class during training, so it misses many instances (low recall of 0.38). This is a **class imbalance problem** — techniques like SMOTE or class weighting would help.

### Metric Reference

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| **Accuracy** | Correct / Total | Overall correctness |
| **Precision** | TP / (TP + FP) | Of predicted attacks, how many were real attacks |
| **Recall** | TP / (TP + FN) | Of real attacks, how many were caught |
| **F1-Score** | 2 × (P × R) / (P + R) | Harmonic mean of precision and recall |

`average='weighted'` weights each class by its support, appropriate given the strong imbalance between Normal/DoS (tens of thousands) and Privilege (tens of samples).

### Evaluation on the Test Set

```python
test_multi_predictions = rf_model_multi.predict(test_X)

print(f"Accuracy:  {accuracy_score(test_y, test_multi_predictions):.4f}")
print(f"Precision: {precision_score(test_y, test_multi_predictions, average='weighted'):.4f}")
print(f"Recall:    {recall_score(test_y, test_multi_predictions, average='weighted'):.4f}")
print(f"F1-Score:  {f1_score(test_y, test_multi_predictions, average='weighted'):.4f}")
```

```
Accuracy:  0.9949
Precision: 0.9947
Recall:    0.9949
F1-Score:  0.9947
```

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Normal | 0.99 | 1.00 | 1.00 | 15,402 |
| DoS | 1.00 | 1.00 | 1.00 | 10,721 |
| Probe | 0.99 | 1.00 | 1.00 | 2,796 |
| Privilege | 0.62 | 0.24 | 0.34 | 21 |
| Access | 0.96 | 0.92 | 0.94 | 764 |

![Test Confusion Matrix](Test-Confusion-Matrix.png)

The test set results closely mirror validation — the model generalizes well. Privilege Escalation remains the weak point due to its tiny representation in the dataset.

---

## 6. Saving and Loading the Model

```python
import joblib

model_filename = 'network_anomaly_detection_model.joblib'
joblib.dump(rf_model_multi, model_filename)
print(f"Model saved to {model_filename}")
```

`joblib.dump` serializes the entire `RandomForestClassifier` — all 100 decision trees, their split thresholds, and class mappings — to disk.

### Loading and Predicting

```python
loaded_model = joblib.load('network_anomaly_detection_model.joblib')
predictions = loaded_model.predict(new_features)
```

Loading is instant — no retraining required.

> **Important:** Unlike the Spam-Classification pipeline (which wraps the vectorizer inside a sklearn `Pipeline`), feature engineering here happens in plain pandas code outside the model. When loading the saved model, you must still manually apply encoding and feature selection before calling `predict`.

The workflow for predicting on new traffic:

1. **Receive** raw connection record
2. **Engineer features** — one-hot encode categoricals, select numeric columns, `reindex` to match training schema
3. **Predict** with `loaded_model.predict(new_features)`
4. **Map** the output integer back to a label: `{0: 'Normal', 1: 'DoS', 2: 'Probe', 3: 'Privilege', 4: 'Access'}`
