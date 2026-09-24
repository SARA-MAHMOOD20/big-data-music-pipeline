"""
Task 2: Distributed Raw Storage (Hadoop HDFS).

Ingests the fma_medium audio + metadata into HDFS via the WebHDFS API
(python `hdfs` client), partitioned by genre:

    /music_pipeline/audio/<genre>/<track_id>.mp3
    /music_pipeline/metadata/*.csv

Reports total ingestion time and the resulting directory/block layout.

Usage:
    python ingest_to_hdfs.py [--webhdfs-url http://localhost:9870] [--workers 8] [--limit N]
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_extraction"))
from utils import load_medium_subset  # noqa: E402

from hdfs import InsecureClient

HERE = os.path.dirname(__file__)


def slugify(genre: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", genre.strip())


def upload_one(client, local_path, hdfs_path):
    if not os.path.exists(local_path):
        return False, "missing"
    try:
        client.upload(hdfs_path, local_path, overwrite=True)
        return True, None
    except Exception as exc:
        return False, str(exc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--webhdfs-url", default=os.environ.get("WEBHDFS_URL", "http://localhost:9870"))
    ap.add_argument("--user", default="root")
    ap.add_argument("--base", default="/music_pipeline")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    client = InsecureClient(args.webhdfs_url, user=args.user)

    df = load_medium_subset()
    if args.limit:
        df = df.head(args.limit)

    client.makedirs(f"{args.base}/audio")
    client.makedirs(f"{args.base}/metadata")

    genres = sorted(df["genre_top"].unique())
    for g in genres:
        client.makedirs(f"{args.base}/audio/{slugify(g)}")

    jobs = []
    for _, row in df.iterrows():
        hdfs_path = f"{args.base}/audio/{slugify(row['genre_top'])}/{row['track_id']}.mp3"
        jobs.append((row["path"], hdfs_path))

    print(f"Uploading {len(jobs)} audio files across {len(genres)} genre partitions -> {args.base}/audio")
    t0 = time.time()
    ok, fail = 0, 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(upload_one, client, lp, hp): (lp, hp) for lp, hp in jobs}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            success, err = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
            if i % 500 == 0 or i == len(jobs):
                elapsed = time.time() - t0
                print(f"[{i}/{len(jobs)}] ok={ok} fail={fail} elapsed={elapsed:.1f}s "
                      f"rate={i/elapsed:.1f}/s", flush=True)
    audio_elapsed = time.time() - t0

    metadata_dir = os.path.join(HERE, "..", "fma_metadata")
    t1 = time.time()
    meta_uploaded = []
    for fname in ["tracks.csv", "genres.csv", "features.csv", "echonest.csv"]:
        local = os.path.join(metadata_dir, fname)
        if os.path.exists(local):
            client.upload(f"{args.base}/metadata/{fname}", local, overwrite=True)
            meta_uploaded.append(fname)
    metadata_elapsed = time.time() - t1

    # Also ingest our extracted feature vectors (Task 1 output) so Spark can
    # read the same feature data from HDFS that spark_mongo_analysis.py reads
    # from MongoDB, for a fair read/processing-time benchmark (Task 4).
    features_dir = os.path.join(HERE, "..", "feature_extraction")
    client.makedirs(f"{args.base}/features")
    for fname in ["features_raw.csv", "features_normalized.csv"]:
        local = os.path.join(features_dir, fname)
        if os.path.exists(local):
            client.upload(f"{args.base}/features/{fname}", local, overwrite=True)
            meta_uploaded.append(fname)

    # Capture the resulting HDFS directory / block layout for the report.
    layout = {}
    for g in genres:
        d = f"{args.base}/audio/{slugify(g)}"
        try:
            status = client.status(d)
            content = client.content(d)
            layout[g] = {
                "path": d,
                "file_count": content.get("fileCount"),
                "space_consumed_bytes": content.get("spaceConsumed"),
            }
        except Exception as exc:
            layout[g] = {"error": str(exc)}

    report = {
        "audio_files_ok": ok,
        "audio_files_failed": fail,
        "audio_ingest_seconds": round(audio_elapsed, 2),
        "metadata_files_uploaded": meta_uploaded,
        "metadata_ingest_seconds": round(metadata_elapsed, 2),
        "total_seconds": round(audio_elapsed + metadata_elapsed, 2),
        "genre_partitions": layout,
    }
    report_path = os.path.join(HERE, "ingestion_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nDone. ok={ok} fail={fail} audio_time={audio_elapsed:.1f}s "
          f"metadata_time={metadata_elapsed:.1f}s")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
