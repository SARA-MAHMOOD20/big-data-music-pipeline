"""
Task 4 (HDFS half): Distributed Analysis with Spark, reading the same feature
data from HDFS instead of MongoDB, for a direct read/processing-time
benchmark against spark_mongo_analysis.py.

Usage:
    python spark_hdfs_analysis.py [--hdfs-url hdfs://localhost:9000]
"""
import argparse
import json
import os
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

HERE = os.path.dirname(__file__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdfs-url", default=os.environ.get("HDFS_URL", "hdfs://localhost:9000"))
    ap.add_argument("--path", default="/music_pipeline/features/features_raw.csv")
    ap.add_argument("--master", default=os.environ.get("SPARK_MASTER", "local[*]"))
    args = ap.parse_args()

    t_start = time.time()
    spark = SparkSession.builder.appName("HDFSAnalysis").master(args.master).getOrCreate()
    df = spark.read.option("header", True).option("inferSchema", True).csv(f"{args.hdfs_url}{args.path}")
    df.cache()
    n = df.count()
    t_read = time.time()

    print(f"Loaded {n} rows from HDFS in {t_read - t_start:.2f}s")

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

    corr = df.stat.corr("tempo", "spectral_centroid_mean")
    print(f"Correlation(tempo, spectral_centroid_mean) = {corr:.4f}")

    q1, q3 = df.approxQuantile("tempo", [0.25, 0.75], 0.01)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = df.filter((F.col("tempo") < lower) | (F.col("tempo") > upper))
    n_outliers = outliers.count()
    print(f"Tempo IQR=[{q1:.1f}, {q3:.1f}], outlier bounds=[{lower:.1f}, {upper:.1f}], "
          f"outliers found: {n_outliers}")

    t_end = time.time()

    report = {
        "source": "hdfs",
        "rows": n,
        "read_seconds": round(t_read - t_start, 3),
        "total_processing_seconds": round(t_end - t_start, 3),
        "tempo_spectral_centroid_correlation": round(corr, 4),
        "tempo_outliers": n_outliers,
    }
    with open(os.path.join(HERE, "benchmark_hdfs.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

    spark.stop()


if __name__ == "__main__":
    main()
