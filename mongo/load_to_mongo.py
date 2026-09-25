"""
Task 3: Feature Storage (MongoDB).

Batch-loads raw + normalized features from CSV into MongoDB, following the
schema documented in schema.md. Inserts in batches (not one document at a
time) via insert_many / bulk_write.

Usage:
    python load_to_mongo.py [--uri mongodb://localhost:27017] [--batch-size 1000]
"""
import argparse
import math
import os
import time

import pandas as pd
from pymongo import MongoClient, ASCENDING
from pymongo.operations import ReplaceOne

HERE = os.path.dirname(__file__)
MFCC_N = 13
CHROMA_N = 12
CONTRAST_N = 7


def row_to_features(row, prefix=""):
    return {
        "mfcc_mean": [row[f"{prefix}mfcc_{i}_mean"] for i in range(MFCC_N)],
        "mfcc_std": [row[f"{prefix}mfcc_{i}_std"] for i in range(MFCC_N)],
        "spectral_centroid_mean": row[f"{prefix}spectral_centroid_mean"],
        "spectral_centroid_std": row[f"{prefix}spectral_centroid_std"],
        "spectral_rolloff_mean": row[f"{prefix}spectral_rolloff_mean"],
        "spectral_rolloff_std": row[f"{prefix}spectral_rolloff_std"],
        "chroma_mean": [row[f"{prefix}chroma_{i}_mean"] for i in range(CHROMA_N)],
        "chroma_std": [row[f"{prefix}chroma_{i}_std"] for i in range(CHROMA_N)],
        "zcr_mean": row[f"{prefix}zcr_mean"],
        "zcr_std": row[f"{prefix}zcr_std"],
        "tempo": row[f"{prefix}tempo"],
        "spectral_bandwidth_mean": row[f"{prefix}spectral_bandwidth_mean"],
        "spectral_bandwidth_std": row[f"{prefix}spectral_bandwidth_std"],
        "contrast_mean": [row[f"{prefix}contrast_{i}_mean"] for i in range(CONTRAST_N)],
        "contrast_std": [row[f"{prefix}contrast_{i}_std"] for i in range(CONTRAST_N)],
        "rms_mean": row[f"{prefix}rms_mean"],
        "rms_std": row[f"{prefix}rms_std"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default=os.environ.get("MONGO_URI", "mongodb://localhost:27017"))
    ap.add_argument("--db", default="music_pipeline")
    ap.add_argument("--collection", default="tracks")
    ap.add_argument("--raw", default=os.path.join(HERE, "..", "feature_extraction", "features_raw.csv"))
    ap.add_argument("--normalized", default=os.path.join(HERE, "..", "feature_extraction", "features_normalized.csv"))
    ap.add_argument("--batch-size", type=int, default=1000)
    args = ap.parse_args()

    raw = pd.read_csv(args.raw).set_index("track_id")
    norm = pd.read_csv(args.normalized).set_index("track_id")

    client = MongoClient(args.uri)
    coll = client[args.db][args.collection]
    coll.create_index([("genre_top", ASCENDING)])
    coll.create_index([("features.tempo", ASCENDING)])

    track_ids = raw.index.tolist()
    n_batches = math.ceil(len(track_ids) / args.batch_size)
    t0 = time.time()
    total = 0

    for b in range(n_batches):
        batch_ids = track_ids[b * args.batch_size:(b + 1) * args.batch_size]
        ops = []
        for tid in batch_ids:
            r = raw.loc[tid]
            doc = {
                "_id": int(tid),
                "genre_top": r["genre_top"],
                "split": r["split"],
                "features": row_to_features(r),
            }
            if tid in norm.index:
                doc["features_normalized"] = row_to_features(norm.loc[tid])
            ops.append(ReplaceOne({"_id": int(tid)}, doc, upsert=True))
        if ops:
            coll.bulk_write(ops, ordered=False)
            total += len(ops)
        if (b + 1) % 5 == 0 or b == n_batches - 1:
            print(f"[batch {b+1}/{n_batches}] inserted={total} elapsed={time.time()-t0:.1f}s", flush=True)

    print(f"Done. {total} documents upserted into {args.db}.{args.collection} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
