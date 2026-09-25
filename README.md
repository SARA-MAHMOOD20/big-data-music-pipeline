# Music Feature Analysis & Real-Time Recommendation System

End-to-end big-data pipeline on the FMA `fma_medium` dataset (25,000 tracks,
16 genres): Librosa feature extraction -> HDFS raw storage -> MongoDB feature
store -> Spark distributed analysis -> genre classifier -> Annoy
recommender -> Kafka real-time streaming -> a containerized Flask web app.

Run end-to-end on the full dataset (24,985/25,000 tracks decoded
successfully): 65.6% genre-classifier test accuracy, 25,000 files ingested
into HDFS in 7.2 minutes, Kafka crash-and-resume demonstrated with zero
data loss across 500 streamed events. Full numbers in
[`report/report.md`](report/report.md).

## Architecture

```
                        +-------------+      +-------------------+
  fma_medium/     ----> | HDFS        |      | features_raw.csv  |
  fma_metadata/         | (namenode/  |<-----| features_norm.csv |
       |                |  datanode)  |      +--------+----------+
       v                 +-----+-----+                |
  extract_features.py          |                       v
  (Librosa: MFCC, centroid,    |                 load_to_mongo.py
   rolloff, chroma, ZCR,       |                       |
   tempo, bandwidth,           v                       v
   contrast, RMS)       spark_hdfs_analysis.py  +-------------+
       |                      \                 | MongoDB     |
       v                       \                | tracks coll.|
  normalize_features.py         \               +------+------+
       |                         \                     |
       v                   spark_mongo_analysis.py <----+
  train_classifier.py             \
  build_index.py (Annoy)           v
       |                    compare_benchmarks.py
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
`docker/docker-compose.yml` and starts with one command
(`docker compose up -d`) - verified to come back healthy with all MongoDB/
HDFS data intact after a full `docker compose down` + `up -d` cycle (named
volumes persist container recreation).

A `jobs` service (`docker/jobs.Dockerfile`) is also defined: WebHDFS
uploads/reads redirect to the datanode's internal container hostname,
which a client on the host machine can't resolve, so the two scripts that
touch HDFS directly (`hdfs_ingest/ingest_to_hdfs.py`,
`spark_analysis/spark_hdfs_analysis.py`) run inside `jobs` instead of on
the host - see the task table below.

## Setup

1. **Get the data** (not committed to git - see `.gitignore`):
   - `fma_metadata.zip` -> extract to `fma_metadata/` (tracks.csv, genres.csv, features.csv, echonest.csv)
   - `fma_medium.zip` -> extract to `fma_medium/` (25,000 mp3s, partitioned in 000-155 subfolders)
   - Source: https://github.com/tashi-2004/FMA-A-Dataset-For-Music-Analysis

2. **Python environment**: `pip install -r requirements.txt` (Python 3.11 recommended -
   librosa's numba/llvmlite dependency does not yet have prebuilt wheels for
   very new Python versions).

3. **Start the whole stack** (one command):
   ```bash
   cd docker
   docker compose up -d
   ```
   Wait ~30s for namenode/datanode health checks to pass (`docker compose ps`).
   This also builds and starts the web app at http://localhost:5000 (it will
   report "models not loaded" until you've run the training steps below at
   least once and populated `models/`).

## Running each task

Commands marked **[jobs]** run inside the `jobs` container
(`docker compose run --rm jobs <command>`, from the `docker/` directory)
because they need to resolve HDFS's internal container hostnames. Everything
else runs on the host with the Python env active (`pip install -r requirements.txt`,
Python 3.11 recommended - librosa's numba/llvmlite dependency has no
prebuilt wheels yet for very new Python versions).

| # | Task | Command |
|---|------|---------|
| 1 | Feature extraction | `python feature_extraction/extract_features.py` |
| 1 | Normalization | `python feature_extraction/normalize_features.py` |
| 1 | Genre distribution plots | `python feature_extraction/plot_distributions.py` |
| 2 | HDFS ingestion **[jobs]** | `python hdfs_ingest/ingest_to_hdfs.py --workers 24` |
| 3 | Load MongoDB | `python mongo/load_to_mongo.py` |
| 3 | Query demo | `python mongo/queries.py` |
| 4 | Spark + Mongo analysis | `python spark_analysis/spark_mongo_analysis.py` |
| 4 | Spark + HDFS analysis **[jobs]** | `python spark_analysis/spark_hdfs_analysis.py` |
| 4 | Compare benchmarks | `python spark_analysis/compare_benchmarks.py` |
| 5 | Train genre classifier | `python classifier/train_classifier.py` |
| 6 | Build Annoy index | `python recommender/build_index.py` |
| 6 | Evaluate precision@5 | `python recommender/evaluate.py` |
| 7 | Kafka producer | `python kafka/producer.py` |
| 7 | Kafka consumer | `python kafka/consumer.py` |
| 8 | Web app (dev, outside Docker) | `python webapp/app.py` -> http://localhost:5000 |
| 8 | Web app (containerized) | included in `docker compose up -d` above; rebuild after changing code with `docker compose up -d --build webapp` |

Feature extraction and HDFS ingestion write resumable checkpoints (rerunning
skips already-processed tracks), since a 25,000-track run can take a while
(~50-75 minutes for feature extraction, ~7 minutes for HDFS ingestion) and
is safe to interrupt.

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
- Plain-language explainer PDF (what each technology is, why it was chosen, and
  how it's used here, task by task): [`report/project_explainer.pdf`](report/project_explainer.pdf)
- Screen-recorded walkthrough: _add your recording link here before submission_

## Known limitations

See `report/report.md` "Known limitations" section.
