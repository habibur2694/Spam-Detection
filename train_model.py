import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "spam.csv"
MODEL_PATH = BASE_DIR / "spam_model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"

def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].str.lower().str.strip()
    df["message"] = df["message"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.25,
        random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
    print(classification_report(y_test, predictions, zero_division=0))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved vectorizer to: {VECTORIZER_PATH}")

if __name__ == "__main__":
    main()
