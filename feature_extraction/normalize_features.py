"""
Normalize extracted audio features (Task 1).

Justification for z-score (StandardScaler) over min-max:
Raw features live on wildly different scales and units - spectral centroid/
rolloff are in Hz (hundreds to thousands), chroma and zero-crossing-rate are
bounded in [0, 1], MFCCs range roughly [-300, 150], and tempo is in BPM
(~50-250). Any distance-based downstream step (Annoy nearest-neighbour
search, k-NN style evaluation, SVM) computes Euclidean/angular distance over
the raw vector, so an unscaled feature like spectral centroid (values in the
thousands) would dominate the distance purely because of its unit, not its
actual discriminative power. Z-score standardization (mean 0, std 1) puts
every feature on a comparable scale. Min-max was considered but rejected:
FMA tracks include some near-silent or clipped outliers, and min-max scaling
is sensitive to those extremes, compressing the bulk of the distribution
into a narrow sub-range. Z-score is far more robust to a handful of outlier
tracks since it centers on the mean/std rather than the min/max.

Usage:
    python normalize_features.py [--in features_raw.csv] [--out features_normalized.csv]
"""
import argparse
import os

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

NON_FEATURE_COLS = {"track_id", "genre_top", "split"}


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(__file__)
    ap.add_argument("--in", dest="inp", default=os.path.join(here, "features_raw.csv"))
    ap.add_argument("--out", default=os.path.join(here, "features_normalized.csv"))
    ap.add_argument("--scaler-out", default=os.path.join(here, "..", "models", "scaler.joblib"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.scaler_out), exist_ok=True)
    df = pd.read_csv(args.inp)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    scaler = StandardScaler()
    normalized = scaler.fit_transform(df[feature_cols])

    out = df[["track_id", "genre_top", "split"]].copy()
    norm_df = pd.DataFrame(normalized, columns=feature_cols, index=df.index)
    out = pd.concat([out, norm_df], axis=1)
    out.to_csv(args.out, index=False)
    joblib.dump({"scaler": scaler, "feature_cols": feature_cols}, args.scaler_out)

    print(f"Normalized {len(out)} tracks x {len(feature_cols)} features -> {args.out}")
    print(f"Scaler saved -> {args.scaler_out}")


if __name__ == "__main__":
    main()
