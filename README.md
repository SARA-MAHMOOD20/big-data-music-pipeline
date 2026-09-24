# Music Feature Analysis & Real-Time Recommendation System

End-to-end big-data pipeline on the FMA `fma_medium` dataset (25,000 tracks,
16 genres): Librosa feature extraction -> HDFS raw storage -> MongoDB feature
store -> Spark distributed analysis -> genre classifier -> Annoy
recommender -> Kafka real-time streaming -> a containerized Flask web app.

## Architecture

```
                     +-------------+      +-------------------+
  fma_medium/  ----> | HDFS        |      | features_raw.csv  |
  fma_metadata/      | (namenode/  |<-----| features_norm.csv |
       |             |  datanode)  |      +--------+----------+
       v              +-----+-----+                |
  extract_features.py       |                        v
  (Librosa: MFCC,           |                  load_to_mongo.py
   centroid, rolloff,       |                        |
   chroma, ZCR, tempo)      v                        v
       |             spark_hdfs_analysis.py   +-------------+
       v                    \                  | MongoDB     |
  normalize_features.py      \                 | tracks coll.|
       |                      \                +------+------+
       v                       \                      |
  train_classifier.py    spark_mongo_analysis.py <-----+
  build_index.py (Annoy)        \
       |                         v
       |                  compare_benchmarks.py
       v
  producer.py --> Kafka (track-events) --> consumer.py --> MongoDB
  (streams tracks)                        (Annoy lookup,  (stream_
                                            offsets committed  recommendations)
                                            only after write)
                                                  |
                                                  v
                                         webapp/app.py (Flask)
                                     upload audio -> insights + recs
```

Full stack (Hadoop, MongoDB, Kafka+Zookeeper, Spark, web app) is packaged in
`docker/docker-compose.yml` and starts with one command.

## Setup

1. **Get the data** (not committed to git - see `.gitignore`):
   - `fma_metadata.zip` -> extract to `fma_metadata/` (tracks.csv, genres.csv, features.csv, echonest.csv)
   - `fma_medium.zip` -> extract to `fma_medium/` (25,000 mp3s, partitioned in 000-155 subfolders)
   - Source: https://github.com/tashi-2004/FMA-A-Dataset-For-Music-Analysis

2. **Python environment**: `pip install -r requirements.txt` (Python 3.11 recommended -
   librosa's numba/llvmlite dependency does not yet have prebuilt wheels for
   very new Python versions).

3. **Start the infrastructure**:
   ```bash
   cd docker
   docker compose up -d namenode datanode mongodb zookeeper kafka spark-master spark-worker
   ```
   Wait ~30s for namenode/datanode health checks to pass (`docker compose ps`).

## Running each task

All commands below are run from the repo root with the Python env active.

| # | Task | Command |
|---|------|---------|
| 1 | Feature extraction | `python feature_extraction/extract_features.py` |
| 1 | Normalization | `python feature_extraction/normalize_features.py` |
| 1 | Genre distribution plots | `python feature_extraction/plot_distributions.py` |
| 2 | HDFS ingestion | `python hdfs_ingest/ingest_to_hdfs.py` |
| 3 | Load MongoDB | `python mongo/load_to_mongo.py` |
| 3 | Query demo | `python mongo/queries.py` |
| 4 | Spark + Mongo analysis | `python spark_analysis/spark_mongo_analysis.py` |
| 4 | Spark + HDFS analysis | `python spark_analysis/spark_hdfs_analysis.py` |
| 4 | Compare benchmarks | `python spark_analysis/compare_benchmarks.py` |
| 5 | Train genre classifier | `python classifier/train_classifier.py` |
| 6 | Build Annoy index | `python recommender/build_index.py` |
| 6 | Evaluate precision@5 | `python recommender/evaluate.py` |
| 7 | Kafka producer | `python kafka/producer.py` |
| 7 | Kafka consumer | `python kafka/consumer.py` |
| 8 | Web app (dev) | `python webapp/app.py` -> http://localhost:5000 |
| 8 | Web app (containerized) | `docker compose -f docker/docker-compose.yml up -d --build webapp` |

Feature extraction and HDFS ingestion write resumable checkpoints (rerunning
skips already-processed tracks), since a 25,000-track run can take a while
and is safe to interrupt.

## Kafka fault-tolerance demo (crash + resume)

`kafka/demo_crash_resume.sh` automates the full demonstration:

```bash
cd kafka
./demo_crash_resume.sh 200 40   # 200 events, simulate a crash after 40
```

It: (1) produces 200 track events, (2) starts the consumer with
`--crash-after 40`, which hard-kills itself (`os._exit(1)`, equivalent to
`kill -9`) partway through, (3) prints the offset Kafka committed before the
crash, (4) restarts the consumer, which resumes from that exact offset, and
(5) checks that `stream_recommendations` in MongoDB has exactly one document
per event - proving no event was lost and none was double-processed.

To do it manually instead: run `python kafka/producer.py`, run
`python kafka/consumer.py` in another terminal, `kill -9 <pid>` mid-stream,
then run `python kafka/consumer.py` again and watch it pick up where it left
off (see `kafka/consumer.py` docstring for the offset-commit strategy).

## Report & walkthrough

- Technical report: [`report/report.md`](report/report.md)
- Screen-recorded walkthrough: _add your recording link here before submission_

## Known limitations

See `report/report.md` "Known limitations" section.
