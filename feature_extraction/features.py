"""Shared single-track feature extraction, used by both the batch extraction
script (extract_features.py) and the web app (Task 8) so the two stay in
sync feature-for-feature."""
import numpy as np

SR = 22050
N_MFCC = 13

FEATURE_ORDER = (
    [f"mfcc_{i}_mean" for i in range(N_MFCC)]
    + [f"mfcc_{i}_std" for i in range(N_MFCC)]
    + ["spectral_centroid_mean", "spectral_centroid_std"]
    + ["spectral_rolloff_mean", "spectral_rolloff_std"]
    + [f"chroma_{i}_mean" for i in range(12)]
    + [f"chroma_{i}_std" for i in range(12)]
    + ["zcr_mean", "zcr_std"]
    + ["tempo"]
)


def extract_features_from_signal(y: np.ndarray, sr: int) -> dict:
    """Returns a flat dict of the required Librosa features for one track."""
    import librosa

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    feats = {}
    for i, v in enumerate(mfcc.mean(axis=1)):
        feats[f"mfcc_{i}_mean"] = float(v)
    for i, v in enumerate(mfcc.std(axis=1)):
        feats[f"mfcc_{i}_std"] = float(v)
    feats["spectral_centroid_mean"] = float(centroid.mean())
    feats["spectral_centroid_std"] = float(centroid.std())
    feats["spectral_rolloff_mean"] = float(rolloff.mean())
    feats["spectral_rolloff_std"] = float(rolloff.std())
    for i, v in enumerate(chroma.mean(axis=1)):
        feats[f"chroma_{i}_mean"] = float(v)
    for i, v in enumerate(chroma.std(axis=1)):
        feats[f"chroma_{i}_std"] = float(v)
    feats["zcr_mean"] = float(zcr.mean())
    feats["zcr_std"] = float(zcr.std())
    feats["tempo"] = float(np.atleast_1d(tempo)[0])
    return feats


def extract_features_from_path(path: str):
    import librosa

    y, sr = librosa.load(path, sr=SR, mono=True)
    if y.size == 0:
        return None
    return extract_features_from_signal(y, sr)
