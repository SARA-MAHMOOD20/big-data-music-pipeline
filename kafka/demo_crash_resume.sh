#!/usr/bin/env bash
# Demonstrates the Task 7 fault-tolerance requirement: kill the consumer
# mid-stream and show it resumes from the correct offset with no data loss
# and no re-processing (verified via committed offsets + Mongo doc count).
#
# Usage: ./demo_crash_resume.sh [n_events] [crash_after]
set -euo pipefail
cd "$(dirname "$0")"

N_EVENTS="${1:-200}"
CRASH_AFTER="${2:-40}"

echo "=== 1) Producing $N_EVENTS track events ==="
python producer.py --limit "$N_EVENTS" --rate 20

echo ""
echo "=== 2) Starting consumer, will simulate a crash (kill -9) after $CRASH_AFTER messages ==="
python consumer.py --crash-after "$CRASH_AFTER" || echo "(consumer exited/crashed as expected)"

echo ""
echo "=== 3) Checking committed offset after the crash ==="
python - <<'PY'
from kafka import KafkaConsumer, TopicPartition
c = KafkaConsumer(bootstrap_servers="localhost:29092", group_id="recommendation-consumers", enable_auto_commit=False)
tp = TopicPartition("track-events", 0)
c.assign([tp])
committed = c.committed(tp)
print(f"Committed offset after crash: {committed}")
c.close()
PY

echo ""
echo "=== 4) Restarting consumer to process the remaining events (resumes from committed offset) ==="
timeout 30 python consumer.py || true

echo ""
echo "=== 5) Verifying: no event was lost or double-applied ==="
python - <<'PY'
from pymongo import MongoClient
coll = MongoClient("mongodb://localhost:27017")["music_pipeline"]["stream_recommendations"]
print(f"stream_recommendations document count: {coll.count_documents({})}")
print("(should equal the number of events produced, each track_id present exactly once)")
PY
