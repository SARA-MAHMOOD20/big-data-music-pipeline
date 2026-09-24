"""Shared helpers for locating FMA tracks and their genre labels."""
import os
import pandas as pd

METADATA_DIR = os.path.join(os.path.dirname(__file__), "..", "fma_metadata")
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "..", "fma_medium")


def track_id_to_path(track_id: int, audio_dir: str = AUDIO_DIR) -> str:
    tid_str = "{:06d}".format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + ".mp3")


def load_medium_subset(metadata_dir: str = METADATA_DIR) -> pd.DataFrame:
    """Return tracks.csv rows belonging to the fma_medium subset (small+medium == 25,000 tracks),
    with a flat genre_top column and a resolved audio file path."""
    tracks = pd.read_csv(os.path.join(metadata_dir, "tracks.csv"), index_col=0, header=[0, 1])
    subset = tracks[tracks[("set", "subset")].isin(["small", "medium"])].copy()
    df = pd.DataFrame(index=subset.index)
    df["track_id"] = subset.index
    df["genre_top"] = subset[("track", "genre_top")]
    df["split"] = subset[("set", "split")]
    df = df.dropna(subset=["genre_top"])
    df["path"] = df["track_id"].apply(track_id_to_path)
    return df.reset_index(drop=True)


def load_genres(metadata_dir: str = METADATA_DIR) -> pd.DataFrame:
    return pd.read_csv(os.path.join(metadata_dir, "genres.csv"), index_col=0)
