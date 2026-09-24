"""
Prints a side-by-side benchmark comparison of the MongoDB-backed vs
HDFS-backed Spark analysis runs (Task 4 deliverable: explain the difference).

Usage:
    python compare_benchmarks.py
"""
import json
import os

HERE = os.path.dirname(__file__)


def main():
    with open(os.path.join(HERE, "benchmark_mongo.json")) as f:
        mongo = json.load(f)
    with open(os.path.join(HERE, "benchmark_hdfs.json")) as f:
        hdfs = json.load(f)

    print(f"{'metric':30s} {'mongodb':>15s} {'hdfs':>15s}")
    print(f"{'read_seconds':30s} {mongo['read_seconds']:>15.3f} {hdfs['read_seconds']:>15.3f}")
    print(f"{'total_processing_seconds':30s} {mongo['total_processing_seconds']:>15.3f} "
          f"{hdfs['total_processing_seconds']:>15.3f}")

    faster = "MongoDB" if mongo["read_seconds"] < hdfs["read_seconds"] else "HDFS"
    print(f"\n{faster} had the faster read in this run.")
    print(
        "Expected explanation: MongoDB serves pre-parsed BSON documents over its wire "
        "protocol with server-side indexing, so small/medium result sets return with low "
        "per-document overhead. HDFS read time here includes CSV parsing (schema "
        "inference + text decoding) on every read, and its throughput advantage really "
        "shows on large sequential scans / big files split across blocks and datanodes - "
        "not on small, single-block CSVs read from a single-node pseudo-cluster. At larger "
        "scale (multi-block files, multiple datanodes), HDFS's parallel block reads would "
        "be expected to close or reverse this gap."
    )


if __name__ == "__main__":
    main()
