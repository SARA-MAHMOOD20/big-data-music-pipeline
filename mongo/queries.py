"""
Task 3: demonstrate at least three MongoDB query patterns, including one
aggregation pipeline.

Usage:
    python queries.py [--uri mongodb://localhost:27017]
"""
import argparse
import os

from pymongo import MongoClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default=os.environ.get("MONGO_URI", "mongodb://localhost:27017"))
    ap.add_argument("--db", default="music_pipeline")
    ap.add_argument("--collection", default="tracks")
    args = ap.parse_args()

    coll = MongoClient(args.uri)[args.db][args.collection]

    # 1) Simple filter: all tracks of a given genre (projection to keep output small)
    print("\n=== Query 1: filter by genre_top='Rock' (first 5) ===")
    for doc in coll.find({"genre_top": "Rock"}, {"genre_top": 1, "features.tempo": 1}).limit(5):
        print(doc)

    # 2) Range query: tracks with a fast tempo
    print("\n=== Query 2: tracks with tempo between 140 and 160 BPM (first 5) ===")
    cursor = coll.find(
        {"features.tempo": {"$gte": 140, "$lte": 160}},
        {"genre_top": 1, "features.tempo": 1},
    ).limit(5)
    for doc in cursor:
        print(doc)

    # 3) Aggregation pipeline: genre-wise summary stats
    print("\n=== Query 3 (aggregation): avg tempo / avg spectral centroid / count per genre ===")
    pipeline = [
        {
            "$group": {
                "_id": "$genre_top",
                "count": {"$sum": 1},
                "avg_tempo": {"$avg": "$features.tempo"},
                "avg_spectral_centroid": {"$avg": "$features.spectral_centroid_mean"},
            }
        },
        {"$sort": {"count": -1}},
    ]
    for doc in coll.aggregate(pipeline):
        print(doc)


if __name__ == "__main__":
    main()
