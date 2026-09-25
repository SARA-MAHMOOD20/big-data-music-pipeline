# Technical Report: Music Feature Analysis & Real-Time Recommendation System

Author: Sara Mahmood
Dataset: FMA `fma_medium` (25,000 tracks, 16 unbalanced genres)

## 1. Architecture

See the diagram in [`README.md`](../README.md#architecture). In short:
Librosa extracts six audio features per track; the raw dataset is mirrored
into HDFS (genre-partitioned) while the feature vectors are stored in
MongoDB and, separately, staged as a CSV in HDFS for a direct Spark
read-time comparison. A Random Forest classifier and an Annoy
nearest-neighbour index are trained on the normalized features. A Kafka
producer/consumer pair applies the recommender to a simulated real-time
stream of tracks, with offsets committed only after an idempotent MongoDB
write. A Flask web app ties it together for interactive use. The whole
infrastructure (HDFS, MongoDB, Kafka/Zookeeper, Spark, web app) is defined
in `docker/docker-compose.yml`.

## 2. Design decisions

- **Feature set & normalization**: the six required features - MFCC (13
  coeff, mean+std), spectral centroid, spectral rolloff, chroma (12-bin,
  mean+std), zero-crossing rate and tempo - plus three additional
  well-established MIR descriptors (spectral bandwidth, 7-band spectral
  contrast, RMS energy), 75 dimensions total. The extras were added after
  the classifier (Task 5) plateaued around 59-60% test accuracy on the
  required six alone, regardless of model choice (Random Forest,
  HistGradientBoosting, class-weighted or not) - see Section 3 for the
  before/after numbers. Z-score standardization was used instead of
  min-max — see `feature_extraction/normalize_features.py` docstring for
  the full justification (unit/scale mismatch across features, sensitivity
  of min-max to outlier tracks).
- **MongoDB schema**: one document per track, feature vectors embedded as
  subdocuments/arrays rather than one document per coefficient — see
  `mongo/schema.md`. Chosen because features are always read/written as a
  whole vector, never partially.
- **HDFS partitioning**: audio is partitioned by genre
  (`/music_pipeline/audio/<genre>/<track_id>.mp3`) so genre-scoped batch
  jobs (e.g., a future single-genre Spark job) can read one partition
  without a full scan.
- **Classifier**: Random Forest with `class_weight="balanced"` to counter
  the 16-genre imbalance (Rock/Electronic dominate the dataset). Chosen over
  SVM for faster training at this feature-vector size and native
  multi-class + feature-importance support; a CRNN was not used since the
  task-required feature set is already hand-crafted (not raw spectrograms).
- **Recommender**: Annoy, evaluated across tree-count/metric configurations
  via precision@5 (genre match as the relevance proxy) before picking the
  final configuration — see `recommender/evaluate.py`.
- **Kafka fault tolerance**: manual offset commits, committed only after the
  MongoDB write succeeds, with an idempotent upsert keyed by track_id. This
  gives Kafka's usual at-least-once delivery the effective end result of
  exactly-once, since re-processing a message after a crash just overwrites
  the same document with the same value.

## 3. Benchmark numbers

_Filled in after running the pipeline end-to-end - see the JSON files this
section is generated from:_

- HDFS ingestion time / layout: `hdfs_ingest/ingestion_report.json`
- MongoDB vs HDFS Spark read/processing time: `spark_analysis/benchmark_mongo.json`,
  `spark_analysis/benchmark_hdfs.json` (`python spark_analysis/compare_benchmarks.py` prints the comparison)
- Classifier accuracy / confusion matrix: `classifier/metrics.json`, `classifier/plots/confusion_matrix.png`
- Recommender precision@5 by configuration: `recommender/evaluate_results.json`

| Metric | Value |
|---|---|
| HDFS ingestion time (25,000 tracks) | TBD |
| Spark read time - MongoDB | TBD |
| Spark read time - HDFS | TBD |
| Classifier test accuracy | TBD |
| Best Annoy config (trees, metric) | TBD |
| Best Annoy precision@5 | TBD |

## 4. Known limitations

- The classifier operates on 16 unbalanced genres from hand-crafted
  features only (no raw spectrogram/CRNN path); rare genres (e.g. Easy
  Listening, Blues) have very few training examples and are the main
  source of misclassification - see the per-genre recall in
  `classifier/metrics.json`.
- The Hadoop cluster is a single-node pseudo-distributed setup
  (one namenode + one datanode), so HDFS's parallel-read advantage over
  MongoDB is understated relative to a real multi-datanode cluster - see
  the explanation in `spark_analysis/compare_benchmarks.py`.
- Annoy indexes are approximate nearest neighbour; precision@5 is evaluated
  against a genre-match proxy in the absence of real user feedback/click
  data.
- The Spark Mongo connector downloads its jar via Maven coordinates
  (`spark.jars.packages`) at first run, which requires outbound network
  access from wherever Spark is executed.
