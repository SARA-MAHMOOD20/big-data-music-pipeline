"""
Task 1: Audio Feature Extraction.

Extracts MFCC, Spectral Centroid, Spectral Rolloff, Chroma, Zero-Crossing Rate
and Tempo for every track in the fma_medium subset (25,000 tracks, 16 genres),
in parallel, with resumable checkpointing (safe to Ctrl-C and rerun).

Usage:
    python extract_features.py [--limit N] [--workers K] [--out PATH]
"""
import argparse
import csv
import multiprocessing as mp
import os
import sys
import time

from utils import load_medium_subset
from features import FEATURE_ORDER, SR, extract_features_from_path

FEATURE_COLUMNS = ["track_id", "genre_top", "split"] + FEATURE_ORDER


def extract_one(row):
    track_id, genre_top, split, path = row
    if not os.path.exists(path):
        return None
    try:
        feats = extract_features_from_path(path)
        if feats is None:
            return None
        return [track_id, genre_top, split] + [feats[c] for c in FEATURE_ORDER]
    except Exception as exc:  # corrupt / unreadable audio
        sys.stderr.write(f"[skip] track {track_id}: {exc}\n")
        return None


def already_done(out_path):
    done = set()
    if os.path.exists(out_path):
        with open(out_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for r in reader:
                if r:
                    done.add(int(r[0]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "features_raw.csv"))
    args = ap.parse_args()

    df = load_medium_subset()
    if args.limit:
        df = df.head(args.limit)

    done = already_done(args.out)
    todo = df[~df["track_id"].isin(done)]
    print(f"Total in medium subset: {len(df)} | already extracted: {len(done)} | remaining: {len(todo)}")

    write_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    rows = list(zip(todo["track_id"], todo["genre_top"], todo["split"], todo["path"]))

    t0 = time.time()
    n_ok, n_fail = 0, 0
    with open(args.out, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(FEATURE_COLUMNS)
            f.flush()

        with mp.Pool(args.workers) as pool:
            for i, result in enumerate(pool.imap_unordered(extract_one, rows, chunksize=8), 1):
                if result is not None:
                    writer.writerow(result)
                    n_ok += 1
                else:
                    n_fail += 1
                if i % 200 == 0 or i == len(rows):
                    f.flush()
                    elapsed = time.time() - t0
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (len(rows) - i) / rate if rate > 0 else float("inf")
                    print(f"[{i}/{len(rows)}] ok={n_ok} fail={n_fail} "
                          f"rate={rate:.2f}/s eta={eta/60:.1f}min", flush=True)

    print(f"Done. ok={n_ok} fail={n_fail} total_time={(time.time()-t0)/60:.1f}min -> {args.out}")


if __name__ == "__main__":
    main()
