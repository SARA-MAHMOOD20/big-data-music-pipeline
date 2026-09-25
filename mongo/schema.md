# MongoDB Schema - `music_pipeline.tracks`

One document per track, embedding both raw and normalized feature vectors
plus the metadata needed for querying without a join (denormalized on
purpose - reads are far more frequent than writes here, and the feature set
per track is small/fixed size).

```json
{
  "_id": 2,                       // track_id (int), used directly as Mongo _id
  "genre_top": "Hip-Hop",
  "split": "training",
  "features": {
    "mfcc_mean": [13 floats], "mfcc_std": [13 floats],
    "spectral_centroid_mean": 2103.4, "spectral_centroid_std": 512.1,
    "spectral_rolloff_mean": 4210.9, "spectral_rolloff_std": 890.2,
    "chroma_mean": [12 floats], "chroma_std": [12 floats],
    "zcr_mean": 0.081, "zcr_std": 0.021,
    "tempo": 128.0,
    "spectral_bandwidth_mean": 1805.2, "spectral_bandwidth_std": 402.6,
    "contrast_mean": [7 floats], "contrast_std": [7 floats],
    "rms_mean": 0.14, "rms_std": 0.05
  },
  "features_normalized": { /* same shape, z-scored */ }
}
```

Indexes:
- `{ genre_top: 1 }` - genre filtering is the most common query pattern.
- `{ "features.tempo": 1 }` - supports tempo range queries.

Design notes:
- `_id` is set to the FMA `track_id` directly (skip a redundant field + index).
- Feature vectors are stored as embedded arrays/subdocuments rather than one
  document per (track, feature) pair, since features are always read/written
  together as a whole vector - no query needs a single MFCC coefficient in
  isolation.
