"""
Plot and compare feature distributions across genres (Task 1 requirement).
Produces boxplots for a representative subset of features across the
top genres by track count (at least 4, as required).

Usage:
    python plot_distributions.py [--n-genres 4]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

HERE = os.path.dirname(__file__)
PLOT_FEATURES = [
    "tempo",
    "spectral_centroid_mean",
    "spectral_rolloff_mean",
    "zcr_mean",
    "mfcc_0_mean",
    "chroma_0_mean",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.path.join(HERE, "features_raw.csv"))
    ap.add_argument("--n-genres", type=int, default=4)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "plots"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.inp)

    top_genres = df["genre_top"].value_counts().head(args.n_genres).index.tolist()
    sub = df[df["genre_top"].isin(top_genres)]
    print(f"Comparing genres: {top_genres} ({len(sub)} tracks)")

    sns.set_theme(style="whitegrid")
    for feat in PLOT_FEATURES:
        if feat not in sub.columns:
            continue
        plt.figure(figsize=(7, 4.5))
        sns.boxplot(data=sub, x="genre_top", y=feat, order=top_genres)
        plt.title(f"{feat} distribution by genre")
        plt.xlabel("Genre")
        plt.ylabel(feat)
        plt.tight_layout()
        out_path = os.path.join(args.out_dir, f"{feat}_by_genre.png")
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
