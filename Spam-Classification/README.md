# Spam Classification

SMS spam detection using **Multinomial Naive Bayes** with NLP preprocessing.

Built as part of the [Hack The Box Academy](https://academy.hackthebox.com/) **AI Red Teamer** path.

## Overview

A complete ML pipeline that classifies SMS messages as spam or ham (legitimate) using natural language processing and a Naive Bayes classifier. The model achieves high-confidence predictions on unseen messages after being trained on the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) dataset (5,572 messages).

## Pipeline

```
Raw SMS → Lowercase → Remove Punctuation/Numbers → Tokenize → Remove Stop Words → Stem → Vectorize (BoW + Bigrams) → Classify (Naive Bayes)
```

## Tech Stack

- **Python** — pandas, NumPy, scikit-learn, NLTK
- **Model** — Multinomial Naive Bayes with Laplace smoothing (`alpha=0.25`)
- **Features** — Bag of Words with unigrams + bigrams via `CountVectorizer`
- **Tuning** — GridSearchCV with 5-fold cross-validation, optimized for F1 score

## Results

| Message | Prediction | Confidence |
|---------|-----------|------------|
| "Congratulations! You've won a $1000 Walmart gift card..." | Spam | 100% |
| "Hey, are we still meeting up for lunch today?" | Ham | 100% |
| "Urgent! Your account has been compromised..." | Spam | 96% |
| "Reminder: Your appointment is scheduled for tomorrow at 10am." | Ham | 100% |
| "FREE entry in a weekly competition to win an iPad..." | Spam | 100% |

## Quick Start

```bash
pip install -r requirements.txt
```

Open `Spam-Detection.ipynb` and run all cells. The notebook downloads the dataset automatically.

## Learn More

For a detailed walkthrough of the theory (Bayes' theorem, NLP preprocessing, feature extraction) and step-by-step code explanations, see the [WALKTHROUGH.md](./WALKTHROUGH.md).
