"""
Task 6: Recommendation Engine (Annoy) - index build.

Builds an Annoy approximate-nearest-neighbour index over the normalized
feature vectors and saves it alongside a track_id/genre lookup table so
recommend.py and the web app can query it.

Usage:
    python build_index.py [--n-trees 100] [--metric angular]
"""
import argparse
import json
import os

import pandas as pd
from annoy import AnnoyIndex

HERE = os.path.dirname(__file__)
NON_FEATURE_COLS = {"track_id", "genre_top", "split"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in", dest="inp",
        default=os.path.join(HERE, "..", "feature_extraction", "features_normalized.csv"),
    )
    ap.add_argument("--n-trees", type=int, default=100)
    ap.add_argument("--metric", default="angular", choices=["angular", "euclidean", "manhattan"])
    ap.add_argument("--out", default=os.path.join(HERE, "..", "models", "annoy_index.ann"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    df = pd.read_csv(args.inp)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    dim = len(feature_cols)

    index = AnnoyIndex(dim, args.metric)
    for i, row in df.iterrows():
        index.add_item(i, row[feature_cols].values.tolist())
    index.build(args.n_trees)
    index.save(args.out)

    lookup = df[["track_id", "genre_top"]].to_dict(orient="index")
    meta_path = os.path.join(os.path.dirname(args.out), "index_meta.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "feature_cols": feature_cols,
                "dim": dim,
                "metric": args.metric,
                "n_trees": args.n_trees,
                "n_items": len(df),
                "lookup": lookup,
            },
            f,
        )

    print(f"Built Annoy index: {len(df)} items, dim={dim}, metric={args.metric}, "
          f"n_trees={args.n_trees} -> {args.out}")
    print(f"Lookup metadata -> {meta_path}")


if __name__ == "__main__":
    main()
