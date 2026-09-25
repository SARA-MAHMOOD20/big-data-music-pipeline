# Technical Report: Music Feature Analysis & Real-Time Recommendation System

Author: Sara Mahmood
Dataset: FMA `fma_medium` (25,000 tracks, 16 unbalanced genres)

## 1. Architecture

See the diagram in [`README.md`](../README.md#architecture). In short:
Librosa extracts nine audio features per track (the six required plus
three added to reach the accuracy target - see Section 2); the raw dataset
is mirrored into HDFS (genre-partitioned) while the feature vectors are
stored in MongoDB and, separately, staged as a CSV in HDFS for a direct
Spark read-time comparison. An XGBoost classifier and an Annoy
nearest-neighbour index are trained on the normalized features. A Kafka
producer/consumer pair applies the recommender to a simulated real-time
stream of tracks, with offsets committed only after an idempotent MongoDB
write. A Flask web app ties it together for interactive use. The whole
infrastructure (HDFS, MongoDB, Kafka/Zookeeper, Spark, web app) is defined
in `docker/docker-compose.yml` and was verified end-to-end: a full
`docker compose down` + `docker compose up -d` brings back all 8 services
healthy with MongoDB and HDFS data intact (named volumes persist across
container recreation).

## 2. Design decisions

- **Feature set & normalization**: the six required features - MFCC (13
  coeff, mean+std), spectral centroid, spectral rolloff, chroma (12-bin,
  mean+std), zero-crossing rate and tempo - plus three additional
  well-established MIR descriptors (spectral bandwidth, 7-band spectral
  contrast, RMS energy), 75 dimensions total. The extras were added after
  the classifier (Task 5) capped out at 58.6% test accuracy on the
  required six alone (Random Forest, class-weighted) - see Section 3 for
  the before/after numbers. Z-score standardization was used instead of
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
- **Classifier**: tuned XGBoost (`n_estimators=700, max_depth=8,
  learning_rate=0.07, subsample=0.8, colsample_bytree=0.7`). This was
  reached empirically, not as a first choice: Random Forest (both
  `class_weight="balanced"` and unweighted), ExtraTrees,
  HistGradientBoosting, an MLP, and a soft-voting ensemble of several of
  these were all tried on the full 75-feature, 24,985-track set first and
  plateaued between 58.6% and 63.0% test accuracy regardless of tuning -
  class-weighting in particular *reduced* raw accuracy by shifting the
  model's attention toward rare classes at the expense of the two dominant
  ones (Rock, Electronic - 54% of the dataset combined). XGBoost was the
  only approach that reliably cleared 65%, and was verified stable across
  three random train/test splits (64.7-65.8%) before being fixed to
  `random_state=42` for the reported run. A CRNN (the task's third listed
  option) was not attempted - it would require operating on raw
  mel-spectrograms rather than the hand-crafted feature vectors Task 1
  specifies, a materially different pipeline.
- **Recommender**: Annoy, evaluated across tree-count/metric configurations
  via precision@5 (genre match as the relevance proxy) before picking the
  final configuration — see `recommender/evaluate.py`.
- **Kafka fault tolerance**: manual offset commits, committed only after the
  MongoDB write succeeds, with an idempotent upsert keyed by track_id. This
  gives Kafka's usual at-least-once delivery the effective end result of
  exactly-once, since re-processing a message after a crash just overwrites
  the same document with the same value.

## 3. Benchmark numbers

All numbers below are from the real, full-scale run (24,985 of 25,000
fma_medium tracks - 15 failed to decode, see Section 4). Source data:

- HDFS ingestion: `hdfs_ingest/ingestion_report.json`
- MongoDB vs HDFS Spark read/processing time: `spark_analysis/benchmark_mongo.json`,
  `spark_analysis/benchmark_hdfs.json` (`python spark_analysis/compare_benchmarks.py` prints the comparison)
- Classifier accuracy / confusion matrix: `classifier/metrics.json`, `classifier/plots/confusion_matrix.png`
- Recommender precision@5 by configuration: `recommender/evaluate_results.json`

| Metric | Value |
|---|---|
| HDFS ingestion time (25,000 audio files, genre-partitioned) | 434.3s (57.6 files/s, 24 parallel WebHDFS uploads) |
| Spark read time - MongoDB (24,985 docs) | 13.46s |
| Spark read time - HDFS (24,985-row CSV) | 8.76s (HDFS faster here - see explanation below) |
| Spark total processing time - MongoDB | 14.61s |
| Spark total processing time - HDFS | 10.46s |
| Tempo vs spectral-centroid correlation | 0.011 (negligible linear correlation) |
| Tempo outliers (IQR method) | 151 / 24,985 (0.6%) |
| Classifier test accuracy (XGBoost, 75 features) | **65.60%** (19,988 train / 4,997 test) |
| Classifier test accuracy, required-6-features-only baseline | 58.6% (Random Forest) |
| Best Annoy config | 100 trees, angular distance |
| Best Annoy precision@5 | 0.488 (vs 0.456 for 10 trees/angular - see table below) |
| Kafka crash/resume demo | 500 events, crash at offset 150, resumed cleanly, 500/500 documents in `stream_recommendations`, zero loss/duplication |

Annoy configuration comparison (precision@5, 800-track evaluation sample,
genre-match relevance proxy):

| Trees | Metric | Precision@5 |
|---|---|---|
| 10 | angular | 0.456 |
| 50 | angular | 0.479 |
| 50 | euclidean | 0.484 |
| 100 | angular | **0.488** |

HDFS beat MongoDB on raw read time in this run - see
`spark_analysis/compare_benchmarks.py`'s printed
explanation: the Mongo Spark Connector pays a per-document BSON
deserialization + schema-sampling cost that a single-pass flat-CSV read on
HDFS doesn't, at this row count on a single-node cluster. MongoDB's actual
advantage shows up in Task 3's indexed queries (`mongo/queries.py`), which
touch a handful of documents rather than scanning all 24,985.

## 4. Known limitations

- 15 of 25,000 fma_medium tracks failed to decode (corrupt/truncated MP3
  frames - mpg123 errors like "dequantization failed") and were skipped
  throughout the pipeline; 24,985 tracks were used everywhere.
- The classifier operates on 16 unbalanced genres from hand-crafted
  features (no raw spectrogram/CRNN path). Even at 65.6% overall accuracy,
  per-genre recall is very uneven: dominant classes do well (Rock 0.85,
  Old-Time/Historic 0.98 recall) while classes with few examples and high
  acoustic overlap with Rock/Electronic essentially fail (Blues, Soul-RnB,
  Easy Listening: 0.00 recall; Pop: 0.07 recall despite 1,186 training
  examples, because Pop's acoustic profile is genuinely closest to Rock and
  Electronic). See the full per-genre table in `classifier/metrics.json`
  and the confusion matrix.
- The Hadoop cluster is a single-node pseudo-distributed setup
  (one namenode + one datanode), so HDFS's block-parallelism advantage
  over MongoDB at real multi-datanode scale isn't exercised here - see
  the explanation in `spark_analysis/compare_benchmarks.py`.
- Annoy indexes are approximate nearest neighbour; precision@5 is evaluated
  against a genre-match proxy (a recommendation counts as "relevant" if it
  shares the query's genre) in the absence of real user feedback/click
  data - true relevance (e.g. "would a listener actually enjoy this") is
  necessarily broader than genre alone.
- The Spark Mongo connector downloads its jar via Maven coordinates
  (`spark.jars.packages`) at first run, which requires outbound network
  access from wherever Spark is executed.
- WebHDFS uploads/reads redirect to the datanode's internal container
  hostname; a client outside the Docker network (e.g. the host machine)
  cannot resolve that hostname, so `hdfs_ingest/ingest_to_hdfs.py` and
  `spark_analysis/spark_hdfs_analysis.py` must run attached to the compose
  network (the `jobs` service - see README) rather than directly on the host.
