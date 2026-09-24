"""
Task 4 (MongoDB half): Distributed Analysis with Spark, reading from MongoDB.

Connects Spark to MongoDB via the Mongo Spark Connector, performs genre-wise
aggregation, feature correlation, and outlier detection (IQR-based) in Spark
DataFrames, and times the read + processing so it can be benchmarked against
the equivalent HDFS-backed run in spark_hdfs_analysis.py.

Usage:
    python spark_mongo_analysis.py [--uri mongodb://localhost:27017]
"""
import argparse
import json
import os
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

HERE = os.path.dirname(__file__)
FEATURE_COLS = [
    "features.spectral_centroid_mean",
    "features.spectral_rolloff_mean",
    "features.zcr_mean",
    "features.tempo",
]


def build_spark(uri, db, collection, master):
    return (
        SparkSession.builder.appName("MongoAnalysis")
        .master(master)
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0")
        .config("spark.mongodb.read.connection.uri", f"{uri}/{db}.{collection}")
        .getOrCreate()
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default=os.environ.get("MONGO_URI", "mongodb://localhost:27017"))
    ap.add_argument("--db", default="music_pipeline")
    ap.add_argument("--collection", default="tracks")
    ap.add_argument("--master", default=os.environ.get("SPARK_MASTER", "local[*]"))
    args = ap.parse_args()

    t_start = time.time()
    spark = build_spark(args.uri, args.db, args.collection, args.master)
    df = spark.read.format("mongodb").load()
    # Flatten the nested `features` struct into top-level columns: some Spark
    # DataFrame stat functions (approxQuantile) fail to resolve dotted nested
    # paths even though groupBy/agg and stat.corr accept them fine.
    df = df.withColumn("tempo", F.col("features.tempo")).withColumn(
        "spectral_centroid_mean", F.col("features.spectral_centroid_mean")
    )
    df.cache()
    n = df.count()
    t_read = time.time()

    print(f"Loaded {n} documents from MongoDB in {t_read - t_start:.2f}s")

    # Genre-wise aggregation
    genre_stats = (
        df.groupBy("genre_top")
        .agg(
            F.count("*").alias("count"),
            F.avg("tempo").alias("avg_tempo"),
            F.avg("spectral_centroid_mean").alias("avg_spectral_centroid"),
            F.stddev("tempo").alias("std_tempo"),
        )
        .orderBy(F.desc("count"))
    )
    genre_stats.show(20, truncate=False)

    # Feature correlation (tempo vs spectral centroid, as an example pair)
    corr = df.stat.corr("tempo", "spectral_centroid_mean")
    print(f"Correlation(tempo, spectral_centroid_mean) = {corr:.4f}")

    # Outlier detection via IQR on tempo
    q1, q3 = df.approxQuantile("tempo", [0.25, 0.75], 0.01)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = df.filter((F.col("tempo") < lower) | (F.col("tempo") > upper))
    n_outliers = outliers.count()
    print(f"Tempo IQR=[{q1:.1f}, {q3:.1f}], outlier bounds=[{lower:.1f}, {upper:.1f}], "
          f"outliers found: {n_outliers}")

    t_end = time.time()

    report = {
        "source": "mongodb",
        "documents": n,
        "read_seconds": round(t_read - t_start, 3),
        "total_processing_seconds": round(t_end - t_start, 3),
        "tempo_spectral_centroid_correlation": round(corr, 4),
        "tempo_outliers": n_outliers,
    }
    with open(os.path.join(HERE, "benchmark_mongo.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

    spark.stop()


if __name__ == "__main__":
    main()
