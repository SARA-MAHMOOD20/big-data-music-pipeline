"""
Task 6: query helper - return the top-5 nearest tracks for a given track_id,
using the Annoy index built by build_index.py.

Usage:
    python recommend.py --track-id 2 [--k 5]
"""
import argparse
import json
import os

from annoy import AnnoyIndex

HERE = os.path.dirname(__file__)


def load_index(index_path=os.path.join(HERE, "..", "models", "annoy_index.ann"),
                meta_path=os.path.join(HERE, "..", "models", "index_meta.json")):
    with open(meta_path) as f:
        meta = json.load(f)
    index = AnnoyIndex(meta["dim"], meta["metric"])
    index.load(index_path)
    id_to_track = {int(k): v for k, v in meta["lookup"].items()}
    track_to_id = {v["track_id"]: int(k) for k, v in id_to_track.items()}
    return index, meta, id_to_track, track_to_id


def recommend_by_track_id(track_id, k=5, index=None, meta=None, id_to_track=None, track_to_id=None):
    if index is None:
        index, meta, id_to_track, track_to_id = load_index()
    if track_id not in track_to_id:
        raise ValueError(f"track_id {track_id} not found in index")
    internal_id = track_to_id[track_id]
    neighbor_ids, distances = index.get_nns_by_item(internal_id, k + 1, include_distances=True)
    results = []
    for nid, dist in zip(neighbor_ids, distances):
        if nid == internal_id:
            continue
        results.append({**id_to_track[nid], "distance": dist})
        if len(results) == k:
            break
    return results


def recommend_by_vector(vector, k=5, index=None, meta=None, id_to_track=None):
    if index is None:
        index, meta, id_to_track, _ = load_index()
    neighbor_ids, distances = index.get_nns_by_vector(vector, k, include_distances=True)
    return [{**id_to_track[nid], "distance": dist} for nid, dist in zip(neighbor_ids, distances)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--track-id", type=int, required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    recs = recommend_by_track_id(args.track_id, args.k)
    print(json.dumps(recs, indent=2))
