"""
Task 7: Real-Time Streaming with Fault Tolerance (Kafka) - consumer.

Consumes track events from `track-events`, applies the Annoy recommender to
each incoming feature vector, and stores the resulting top-5 recommendations
in MongoDB (`stream_recommendations`, upsert by track_id -> idempotent) so a
crash between "processed" and "committed" never produces duplicate/incorrect
output, only at-worst a harmless reprocessed write.

Fault tolerance strategy:
  - `enable_auto_commit=False`: offsets are committed by us, synchronously,
    only *after* the recommendation has been computed and written to Mongo.
  - A single consumer group id ("recommendation-consumers") persists offsets
    on the Kafka broker, so a killed-and-restarted process resumes exactly
    at the last committed offset - no message is skipped, and any message
    reprocessed after a mid-batch crash is a no-op thanks to the idempotent
    upsert.

To demonstrate: start this consumer, let it process a batch of events,
`kill -9` it mid-stream, restart it, and observe (via the printed offsets
and the Mongo document count) that no event is lost or double-applied.

Usage:
    python consumer.py [--bootstrap kafka:9092] [--group recommendation-consumers]
"""
import argparse
import json
import os
import sys
import time

from kafka import KafkaConsumer
from pymongo import MongoClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "recommender"))
from recommend import load_index, recommend_by_vector  # noqa: E402

TOPIC = "track-events"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default=os.environ.get("KAFKA_BOOTSTRAP", "localhost:29092"))
    ap.add_argument("--topic", default=TOPIC)
    ap.add_argument("--group", default="recommendation-consumers")
    ap.add_argument("--mongo-uri", default=os.environ.get("MONGO_URI", "mongodb://localhost:27017"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--crash-after", type=int, default=None,
                     help="Testing aid: os._exit(1) after N processed messages, to simulate a crash.")
    args = ap.parse_args()

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.bootstrap,
        group_id=args.group,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
    )
    mongo = MongoClient(args.mongo_uri)
    out_coll = mongo["music_pipeline"]["stream_recommendations"]

    index, meta, id_to_track, _ = load_index()

    print(f"Consumer started. group={args.group} topic={args.topic}")
    processed = 0
    try:
        for msg in consumer:
            event = msg.value
            track_id = event["track_id"]
            recs = recommend_by_vector(event["features"], k=args.k, index=index, meta=meta, id_to_track=id_to_track)

            out_coll.replace_one(
                {"_id": track_id},
                {
                    "_id": track_id,
                    "genre_top": event["genre_top"],
                    "recommendations": recs,
                    "partition": msg.partition,
                    "offset": msg.offset,
                    "processed_at": time.time(),
                },
                upsert=True,
            )

            # Commit only after the write above has succeeded -> at-least-once
            # delivery with an idempotent sink means "at-least-once" behaves
            # like "exactly-once" from the consumer of stream_recommendations.
            consumer.commit()
            processed += 1

            if processed % 10 == 0:
                print(f"[partition={msg.partition} offset={msg.offset}] "
                      f"processed track_id={track_id} -> {len(recs)} recs "
                      f"(total processed this run: {processed})", flush=True)

            if args.crash_after and processed >= args.crash_after:
                print(f"Simulating crash after {processed} messages (offset={msg.offset}).")
                os._exit(1)  # hard kill, no cleanup - mirrors `kill -9`

    except KeyboardInterrupt:
        print(f"\nShutting down gracefully. processed={processed}")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
