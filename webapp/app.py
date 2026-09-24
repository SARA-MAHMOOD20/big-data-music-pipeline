"""
Task 8: Web Interface.

Flask app where a user uploads an audio file and receives:
  - extracted feature insights (tempo, spectral centroid, etc.)
  - a predicted genre (from the Task 5 classifier)
  - the top-5 recommended tracks (from the Task 6 Annoy index)

Usage (dev): python app.py
"""
import os
import sys
import tempfile

import joblib
import numpy as np
from flask import Flask, render_template, request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "feature_extraction"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "recommender"))
from features import FEATURE_ORDER, extract_features_from_path  # noqa: E402
from recommend import load_index, recommend_by_vector  # noqa: E402

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

_state = {}


def get_state():
    """Lazy-load models on first request so the app still boots even if
    training hasn't produced artifacts yet (index page will just warn)."""
    if "loaded" in _state:
        return _state
    _state["loaded"] = True
    try:
        scaler_bundle = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
        _state["scaler"] = scaler_bundle["scaler"]
        _state["scaler_feature_cols"] = scaler_bundle["feature_cols"]
    except FileNotFoundError:
        _state["scaler"] = None
    try:
        clf_bundle = joblib.load(os.path.join(MODELS_DIR, "genre_classifier.pkl"))
        _state["classifier"] = clf_bundle["model"]
        _state["classifier_feature_cols"] = clf_bundle["feature_cols"]
    except FileNotFoundError:
        _state["classifier"] = None
    try:
        index, meta, id_to_track, track_to_id = load_index(
            index_path=os.path.join(MODELS_DIR, "annoy_index.ann"),
            meta_path=os.path.join(MODELS_DIR, "index_meta.json"),
        )
        _state["index"], _state["meta"], _state["id_to_track"] = index, meta, id_to_track
    except FileNotFoundError:
        _state["index"] = None
    return _state


@app.route("/", methods=["GET"])
def index():
    state = get_state()
    ready = all(state.get(k) is not None for k in ("scaler", "classifier", "index"))
    return render_template("index.html", ready=ready, result=None)


@app.route("/analyze", methods=["POST"])
def analyze():
    state = get_state()
    file = request.files.get("audio")
    if not file or file.filename == "":
        return render_template("index.html", ready=True, error="No file uploaded.", result=None)

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return render_template("index.html", ready=True,
                                error=f"Unsupported file type '{ext}'.", result=None)

    with tempfile.NamedTemporaryFile(suffix=ext, delete=True) as tmp:
        file.save(tmp.name)
        raw_feats = extract_features_from_path(tmp.name)

    if raw_feats is None:
        return render_template("index.html", ready=True,
                                error="Could not extract features from this file.", result=None)

    raw_vector = np.array([[raw_feats[c] for c in state["scaler_feature_cols"]]])
    norm_vector = state["scaler"].transform(raw_vector)[0]

    genre_pred = state["classifier"].predict(norm_vector.reshape(1, -1))[0]
    proba = None
    if hasattr(state["classifier"], "predict_proba"):
        probs = state["classifier"].predict_proba(norm_vector.reshape(1, -1))[0]
        labels = state["classifier"].classes_
        proba = sorted(zip(labels, probs), key=lambda x: -x[1])[:5]

    recs = recommend_by_vector(
        norm_vector.tolist(), k=5,
        index=state["index"], meta=state["meta"], id_to_track=state["id_to_track"],
    )

    insights = {
        "tempo_bpm": round(raw_feats["tempo"], 1),
        "spectral_centroid_hz": round(raw_feats["spectral_centroid_mean"], 1),
        "spectral_rolloff_hz": round(raw_feats["spectral_rolloff_mean"], 1),
        "zero_crossing_rate": round(raw_feats["zcr_mean"], 4),
        "mfcc_0_mean": round(raw_feats["mfcc_0_mean"], 2),
    }

    result = {
        "filename": file.filename,
        "insights": insights,
        "predicted_genre": genre_pred,
        "genre_probabilities": proba,
        "recommendations": recs,
    }
    return render_template("index.html", ready=True, result=result, error=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
