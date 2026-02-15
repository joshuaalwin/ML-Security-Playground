# Spam Classification — Walkthrough

A step-by-step guide covering the theory and implementation behind this SMS spam detection project. If you're here to learn, this is for you.

## Table of Contents

- [1. Theory: Naive Bayes](#1-theory-naive-bayes)
- [2. Preparing the Dataset](#2-preparing-the-dataset)
- [3. Preprocessing Pipeline](#3-preprocessing-pipeline)
- [4. Feature Extraction](#4-feature-extraction)
- [5. Training and Hyperparameter Tuning](#5-training-and-hyperparameter-tuning)
- [6. Inference on New Messages](#6-inference-on-new-messages)
- [7. Saving and Loading the Model](#7-saving-and-loading-the-model)

---

## 1. Theory: Naive Bayes

Naive Bayes is a machine learning algorithm that describes the probability of an event occurring based on prior knowledge. It applies **Bayes' Theorem**:

```
P(A|B) = (P(B|A) * P(A)) / P(B)
```

| Term | Definition |
|------|-----------|
| `P(A\|B)` | Probability of event A occurring, given that B is true |
| `P(B\|A)` | Probability of event B occurring, given that A is true |
| `P(A)` | Prior probability of event A |
| `P(B)` | Prior probability of event B |

### Applied to Spam Detection

In our context, **A = Email is Spam** and **B = Observed features of the email**:

```
P(Spam|Features) = (P(Features|Spam) * P(Spam)) / P(Features)
```

| Component | Name | Meaning |
|-----------|------|---------|
| `P(Spam\|Features)` | **Hypothesis** | Probability of spam given its features |
| `P(Features\|Spam)` | **Likelihood** | Probability of observing these features in a spam email |
| `P(Spam)` | **Prior Probability** | Overall probability an email is spam |
| `P(Features)` | **Marginal Likelihood** | Probability of the features appearing in any email |

**How the model thinks:** It maintains a "spam dictionary" and a "ham dictionary". For each word, it learns how likely that word is in spam vs ham. When a new message comes in, it multiplies (really: adds the log of) those probabilities across all words and picks the class with the higher overall probability.

---

## 2. Preparing the Dataset

### Downloading and Extracting

The dataset is the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — 5,572 SMS messages labeled as `ham` or `spam`.

```python
import requests, zipfile, io

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
response = requests.get(url)

with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    z.extractall("sms_spam_collection")
```

### Loading into Pandas

The file is tab-separated with no header row, so we specify that manually:

```python
import pandas as pd

df = pd.read_csv(
    "sms_spam_collection/SMSSpamCollection",
    sep="\t", header=None, names=["label", "message"]
)
```

> `sep="\t"` tells pandas the columns are separated by tabs. `header=None` means the file has no header row, so we provide column names via `names`.

### Removing Duplicates

```python
print("Duplicate entries:", df.duplicated().sum())  # 403
df = df.drop_duplicates()
```

403 duplicate entries are found and removed to prevent the model from being biased toward repeated messages.

---

## 3. Preprocessing Pipeline

Preprocessing standardizes the text, reduces noise, and extracts meaningful features. Each step builds on the previous one.

### 3.1 Lowercasing

```python
df["message"] = df["message"].str.lower()
```

Ensures the classifier treats words equally regardless of case. Without this, `"Free"`, `"FREE"`, and `"free"` would be treated as three separate features.

### 3.2 Removing Punctuation and Numbers

```python
import re
df["message"] = df["message"].apply(lambda x: re.sub(r"[^a-z\s$!]", "", x))
```

Most punctuation and numbers increase vocabulary size without adding semantic value. However, this is a **task-aware compromise** — we deliberately keep two symbols:

- **`$`** — often indicates money (`"win $500"`, `"cash prize"`), a strong spam signal
- **`!`** — often indicates urgency (`"HURRY!"`, `"WIN NOW!!!"`), another spam signal

### 3.3 Tokenization

```python
from nltk.tokenize import word_tokenize
df["message"] = df["message"].apply(word_tokenize)
```

Splits each message string into a list of individual tokens.

`"free cash now!"` → `["free", "cash", "now", "!"]`

**Why it matters:** Models and subsequent preprocessing steps operate on **tokens**, not raw strings. Tokenization turns unstructured text into a structured list of units that you can filter, transform, and count.

### 3.4 Removing Stop Words

```python
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))
df["message"] = df["message"].apply(
    lambda x: [word for word in x if word not in stop_words]
)
```

Stop words (`"and"`, `"the"`, `"is"`, `"at"`) are extremely frequent but carry little discriminatory power for spam vs ham. Removing them:

- **Reduces noise** in the feature space
- **Shortens** token sequences
- Puts more emphasis on informative words like `"free"`, `"win"`, `"prize"`, `"call"`

### 3.5 Stemming

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
df["message"] = df["message"].apply(
    lambda x: [stemmer.stem(word) for word in x]
)
```

The Porter Stemmer reduces words to their root form: `"running"` → `"run"`, `"studies"` → `"studi"`.

**Why it matters:**
- Clusters variants of a word together (`run`, `running`, `runs` → `run`)
- Reduces vocabulary size
- Helps the model generalize — it learns one feature for the stem instead of separate features for each inflected form

### 3.6 Joining Tokens Back into Strings

```python
df["message"] = df["message"].apply(lambda x: " ".join(x))
```

Many vectorizers (like scikit-learn's `CountVectorizer`) expect raw text strings as input, not lists. This converts each token list back into a single space-separated string.

Each message is now a preprocessed string: lowercase, minimal punctuation, no numbers, stop words removed, stems retained — **ready for vectorization**.

---

## 4. Feature Extraction

Machine learning models cannot work on raw strings. Each message must be represented as a **vector of numbers** (features).

### Bag of Words

The bag-of-words model builds a vocabulary of all unique terms in the training set, then represents each message as counts of how many times each term appears. Word order is ignored — only which words appear and how often matters.

This alone is enough to capture that words like `"free"`, `"prize"`, `"win"`, `"cash"` tend to appear more in spam.

### Why Bigrams?

With unigrams only, each feature is a single word. By adding **bigrams** (two-word phrases), we capture **local word order**:

- Unigrams: `"free"`, `"prize"`, `"trip"`
- Bigrams: `"free prize"`, `"won trip"`, `"spam message"`

`"free prize"` is a much stronger spam signal than `"free"` alone. Bigrams capture short phrases that are very characteristic of spam.

### CountVectorizer Configuration

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))
X = vectorizer.fit_transform(df["message"])
y = df["label"].apply(lambda x: 1 if x == "spam" else 0)
```

**What each parameter does:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `min_df=1` | Keep terms in at least 1 document | Retains all terms (raise to drop rare noise) |
| `max_df=0.9` | Drop terms in >90% of documents | Removes overly common words that don't help classification |
| `ngram_range=(1,2)` | Unigrams + bigrams | Captures both individual words and two-word phrases |

**The process under the hood:**

1. **Tokenization with n-grams** — splits each message into tokens and generates both unigrams and bigrams
2. **Building the vocabulary** — applies `min_df` and `max_df` filters. Words appearing in >90% of documents are dropped since they don't help distinguish spam from ham
3. **Vectorization** — each document becomes a row in a sparse matrix where cell values are term counts

The result: **X** is a sparse numeric feature matrix, **y** contains labels (1=spam, 0=ham) — ready for the classifier.

---

## 5. Training and Hyperparameter Tuning

### Pipeline

A pipeline chains together vectorization and modeling, ensuring the same data transformation is consistently applied:

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("vectorizer", vectorizer),
    ("classifier", MultinomialNB())
])
```

### GridSearchCV

GridSearchCV is a systematic search to find the best hyperparameters:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "classifier__alpha": [0.01, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1")
grid_search.fit(df["message"], y)
best_model = grid_search.best_estimator_
```

**How it works:**

- `cv=5` splits data into 5 folds
- For each candidate `alpha`, it trains on 4 folds, tests on the remaining one, rotates 5 times, and averages the F1 score
- `classifier__alpha` uses the double underscore to target the `alpha` parameter of `MultinomialNB` inside the pipeline

### What is Alpha (Smoothing)?

Alpha is the **Laplace smoothing** parameter. Here's why it matters:

- **Without smoothing:** if a word never appears in spam during training, the probability of spam for any message containing that word becomes exactly **0**. Since Naive Bayes multiplies probabilities, one zero kills the entire prediction.
- **With smoothing (`alpha > 0`):** the model pretends each word has been seen at least a tiny bit in each class, stabilizing probabilities and preventing the "zero-probability" problem.
- **Small alpha** = fits training data more closely (lower bias, higher variance)
- **Large alpha** = more smoothing, more generalization (higher bias, lower variance)

GridSearchCV tries all candidate alpha values and selects the one with the best F1 score.

**Best result:** `alpha = 0.25`

---

## 6. Inference on New Messages

### Preprocessing Function

**Training and inference must use identical preprocessing.** If they don't, the vectorizer sees different tokens and the model breaks.

```python
def preprocess_message(message):
    message = message.lower()
    message = re.sub(r"[^a-z\s$!]", "", message)
    tokens = word_tokenize(message)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)
```

### Vectorization and Prediction

```python
# Preprocess new messages
processed_messages = [preprocess_message(msg) for msg in new_messages]

# Vectorize using the trained vocabulary (no new words are added)
X_new = best_model.named_steps["vectorizer"].transform(processed_messages)

# Predict labels and probabilities
predictions = best_model.named_steps["classifier"].predict(X_new)
prediction_probabilities = best_model.named_steps["classifier"].predict_proba(X_new)
```

- `.transform()` maps cleaned text to a sparse matrix using the **existing vocabulary** — no new words are added
- `.predict()` gives `0` (ham) or `1` (spam)
- `.predict_proba()` gives two numbers per message: probability of ham and probability of spam

### Example Results

For a spammy message like *"Congratulations! You've won a $1000 Walmart gift card..."*, the model has seen similar patterns (`"congratulations"`, `"won"`, `"gift"`, `"$"`, links) mostly in spam during training, so its spam probability goes to ~1.00.

| Message | Prediction | Spam Probability |
|---------|-----------|-----------------|
| "Congratulations! You've won a $1000 Walmart gift card..." | Spam | 1.00 |
| "Hey, are we still meeting up for lunch today?" | Ham | 0.00 |
| "Urgent! Your account has been compromised..." | Spam | 0.96 |
| "Reminder: Your appointment is scheduled for tomorrow at 10am." | Ham | 0.00 |
| "FREE entry in a weekly competition to win an iPad..." | Spam | 1.00 |

---

## 7. Saving and Loading the Model

We use `joblib` to serialize the trained pipeline for later use:

### Saving

```python
import joblib

model_filename = "spam_detection_model.joblib"
joblib.dump(best_model, model_filename)
```

`joblib.dump` writes the entire `best_model` — pipeline, vectorizer, classifier weights, and hyperparameters — to disk as bytes.

### Loading and Predicting

```python
loaded_model = joblib.load("spam_detection_model.joblib")
processed = [preprocess_message(msg) for msg in new_messages]
predictions = loaded_model.predict(processed)
```

The workflow for predicting on new data:

1. **Receive** raw SMS
2. **Preprocess** with `preprocess_message()`
3. **Predict** with `loaded_model.predict([...])`

That's it — no need to retrain.
