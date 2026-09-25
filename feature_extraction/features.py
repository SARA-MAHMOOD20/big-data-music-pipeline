"""Shared single-track feature extraction, used by both the batch extraction
script (extract_features.py) and the web app (Task 8) so the two stay in
sync feature-for-feature."""
import numpy as np

SR = 22050
N_MFCC = 13

N_CONTRAST_BANDS = 6  # librosa default n_bands=6 -> 7 output rows (bands+1)

FEATURE_ORDER = (
    [f"mfcc_{i}_mean" for i in range(N_MFCC)]
    + [f"mfcc_{i}_std" for i in range(N_MFCC)]
    + ["spectral_centroid_mean", "spectral_centroid_std"]
    + ["spectral_rolloff_mean", "spectral_rolloff_std"]
    + [f"chroma_{i}_mean" for i in range(12)]
    + [f"chroma_{i}_std" for i in range(12)]
    + ["zcr_mean", "zcr_std"]
    + ["tempo"]
    # Extra features beyond the required six, added to give the genre
    # classifier (Task 5) enough signal to clear the 65% accuracy bar -
    # MFCC/centroid/rolloff/chroma/ZCR/tempo alone plateaued around 59-60%
    # regardless of model choice (Random Forest, HistGradientBoosting,
    # class-weighted or not - see report.md).
    + ["spectral_bandwidth_mean", "spectral_bandwidth_std"]
    + [f"contrast_{i}_mean" for i in range(N_CONTRAST_BANDS + 1)]
    + [f"contrast_{i}_std" for i in range(N_CONTRAST_BANDS + 1)]
    + ["rms_mean", "rms_std"]
)


def extract_features_from_signal(y: np.ndarray, sr: int) -> dict:
    """Returns a flat dict of the required Librosa features for one track."""
    import librosa

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    # librosa.beat.beat_track's numba-guvectorize DP step segfaults under the
    # numba build in this environment; the autocorrelation-based tempo
    # estimator below covers the "tempo" feature requirement without it.
    tempo = librosa.feature.tempo(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)

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
    feats["spectral_bandwidth_mean"] = float(bandwidth.mean())
    feats["spectral_bandwidth_std"] = float(bandwidth.std())
    for i, v in enumerate(contrast.mean(axis=1)):
        feats[f"contrast_{i}_mean"] = float(v)
    for i, v in enumerate(contrast.std(axis=1)):
        feats[f"contrast_{i}_std"] = float(v)
    feats["rms_mean"] = float(rms.mean())
    feats["rms_std"] = float(rms.std())
    return feats


def extract_features_from_path(path: str):
    import librosa

    y, sr = librosa.load(path, sr=SR, mono=True)
    if y.size == 0:
        return None
    return extract_features_from_signal(y, sr)
