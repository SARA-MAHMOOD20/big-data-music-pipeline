"""
Task 7: Real-Time Streaming with Fault Tolerance (Kafka) - producer.

Streams one "track event" per row of the extracted feature CSV to the
`track-events` topic, simulating tracks arriving for real-time
recommendation. Each event carries track_id, genre_top and the normalized
feature vector needed by the consumer to query the Annoy index directly
(no extra DB round-trip needed in the consumer).

Usage:
    python producer.py [--bootstrap kafka:9092] [--rate 5] [--limit N]
"""
import argparse
import json
import os
import time

import pandas as pd
from kafka import KafkaProducer

HERE = os.path.dirname(__file__)
NON_FEATURE_COLS = {"track_id", "genre_top", "split"}
TOPIC = "track-events"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default=os.environ.get("KAFKA_BOOTSTRAP", "localhost:29092"))
    ap.add_argument("--topic", default=TOPIC)
    ap.add_argument(
        "--in", dest="inp",
        default=os.path.join(HERE, "..", "feature_extraction", "features_normalized.csv"),
    )
    ap.add_argument("--rate", type=float, default=5.0, help="events per second")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    if args.limit:
        df = df.head(args.limit)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: str(k).encode("utf-8"),
        acks="all",
        retries=5,
        linger_ms=10,
    )

    print(f"Producing {len(df)} track events to '{args.topic}' at {args.rate}/s ...")
    delay = 1.0 / args.rate if args.rate > 0 else 0
    sent = 0
    for _, row in df.iterrows():
        event = {
            "track_id": int(row["track_id"]),
            "genre_top": row["genre_top"],
            "features": [float(row[c]) for c in feature_cols],
            "produced_at": time.time(),
        }
        producer.send(args.topic, key=event["track_id"], value=event)
        sent += 1
        if sent % 50 == 0:
            producer.flush()
            print(f"sent {sent}/{len(df)}", flush=True)
        if delay:
            time.sleep(delay)

    producer.flush()
    print(f"Done. sent={sent}")


if __name__ == "__main__":
    main()
