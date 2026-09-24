"""
Task 5: Genre Classification Model.

Trains a Random Forest on the extracted (normalized) audio features to
predict genre_top. Reports accuracy, full confusion matrix, and per-genre
precision/recall on a held-out stratified test split. Does not use the
label itself, track_id, or split as a feature.

Usage:
    python train_classifier.py [--in features_normalized.csv] [--test-size 0.2]
"""
import argparse
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(__file__)
NON_FEATURE_COLS = {"track_id", "genre_top", "split"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in", dest="inp",
        default=os.path.join(HERE, "..", "feature_extraction", "features_normalized.csv"),
    )
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].values
    y = df["genre_top"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    labels = sorted(df["genre_top"].unique())
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, labels=labels, zero_division=0))

    os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Genre confusion matrix (accuracy={acc:.3f})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_path = os.path.join(HERE, "plots", "confusion_matrix.png")
    plt.savefig(cm_path, dpi=120)
    plt.close()

    models_dir = os.path.join(HERE, "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump({"model": clf, "feature_cols": feature_cols, "labels": labels}, os.path.join(models_dir, "genre_classifier.pkl"))

    metrics = {
        "test_accuracy": acc,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "labels": labels,
        "per_genre": {
            lbl: {
                "precision": report[lbl]["precision"],
                "recall": report[lbl]["recall"],
                "f1": report[lbl]["f1-score"],
                "support": report[lbl]["support"],
            }
            for lbl in labels
        },
        "confusion_matrix": cm.tolist(),
    }
    with open(os.path.join(HERE, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model -> {models_dir}/genre_classifier.pkl, metrics -> metrics.json, confusion matrix -> {cm_path}")
    if acc < 0.65:
        print("WARNING: accuracy below the 65% target - see report for discussion.")


if __name__ == "__main__":
    main()
