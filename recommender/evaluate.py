"""
Task 6: precision@5 evaluation, comparing Annoy configurations (number of
trees and distance metric) to justify the final choice used by build_index.py.

Relevance proxy: a recommended neighbour counts as "relevant" to a query
track if it shares the same genre_top label. This is a standard proxy in the
absence of explicit user feedback/click data, and is reasonable here since
genre similarity is exactly what the hand-crafted timbre/rhythm features
(MFCC, chroma, spectral centroid/rolloff, tempo) are expected to capture.

Usage:
    python evaluate.py [--sample-size 500]
"""
import argparse
import itertools
import json
import os
import random
import time

import pandas as pd
from annoy import AnnoyIndex

HERE = os.path.dirname(__file__)
NON_FEATURE_COLS = {"track_id", "genre_top", "split"}
CONFIGS = [
    {"n_trees": 10, "metric": "angular"},
    {"n_trees": 50, "metric": "angular"},
    {"n_trees": 50, "metric": "euclidean"},
    {"n_trees": 100, "metric": "angular"},
]


def precision_at_k(index, genres, sample_idx, k=5):
    hits, total = 0, 0
    for i in sample_idx:
        query_genre = genres[i]
        neighbor_ids = index.get_nns_by_item(i, k + 1)
        neighbor_ids = [n for n in neighbor_ids if n != i][:k]
        hits += sum(1 for n in neighbor_ids if genres[n] == query_genre)
        total += len(neighbor_ids)
    return hits / total if total else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in", dest="inp",
        default=os.path.join(HERE, "..", "feature_extraction", "features_normalized.csv"),
    )
    ap.add_argument("--sample-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    dim = len(feature_cols)
    genres = df["genre_top"].tolist()
    vectors = df[feature_cols].values.tolist()

    random.seed(args.seed)
    sample_idx = random.sample(range(len(df)), min(args.sample_size, len(df)))

    results = []
    for cfg in CONFIGS:
        t0 = time.time()
        index = AnnoyIndex(dim, cfg["metric"])
        for i, v in enumerate(vectors):
            index.add_item(i, v)
        index.build(cfg["n_trees"])
        build_time = time.time() - t0

        t1 = time.time()
        p_at_5 = precision_at_k(index, genres, sample_idx, k=5)
        eval_time = time.time() - t1

        results.append({
            **cfg,
            "precision_at_5": round(p_at_5, 4),
            "build_seconds": round(build_time, 3),
            "eval_seconds": round(eval_time, 3),
        })
        print(f"n_trees={cfg['n_trees']:>3} metric={cfg['metric']:<10} "
              f"precision@5={p_at_5:.4f} build={build_time:.2f}s")

    best = max(results, key=lambda r: r["precision_at_5"])
    print(f"\nBest config: n_trees={best['n_trees']}, metric={best['metric']} "
          f"(precision@5={best['precision_at_5']})")

    with open(os.path.join(HERE, "evaluate_results.json"), "w") as f:
        json.dump({"sample_size": len(sample_idx), "results": results, "best": best}, f, indent=2)


if __name__ == "__main__":
    main()
