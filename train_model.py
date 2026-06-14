"""
Training script for the TrustLens AI fake review detector.
Loads the CSV, cleans the text a bit, trains a TF-IDF + Logistic
Regression model, prints the metrics and saves the model files.
"""

import os
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

DATA_PATH = "data/reviews.csv"
MODEL_DIR = "models"


def clean_text(text):
    # keep it simple: lowercase, remove extra spaces and weird chars
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s!?.,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["review", "label"])
    df["review"] = df["review"].apply(clean_text)
    return df


def build_model(X, y):
    # TF-IDF with unigrams and bigrams. Bigrams help catch repeated
    # marketing-style phrases like "must buy" or "best ever".
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        stop_words="english",
    )
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_vec, y)
    return model, vectorizer


def train():
    print("Loading dataset...")
    df = load_data(DATA_PATH)
    print(f"Total reviews: {len(df)}")
    print(f"Genuine: {(df['label'] == 0).sum()} | Suspicious: {(df['label'] == 1).sum()}")

    X = df["review"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model, vectorizer = build_model(X_train, y_train)
    X_test_vec = vectorizer.transform(X_test)
    preds = model.predict(X_test_vec)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print("\n--- Results on test set ---")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")
    print("\nDetailed report:")
    print(classification_report(y_test, preds, target_names=["Genuine", "Suspicious"]))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))
    print(f"\nSaved model and vectorizer inside the '{MODEL_DIR}/' folder.")


if __name__ == "__main__":
    train()
