from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np

UCI_HAR_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"


def _read_signal_file(path: Path) -> np.ndarray:
    return np.loadtxt(path)


def ensure_uci_har(root: str | Path) -> Path:
    root = Path(root)
    dataset_dir = root / "UCI HAR Dataset"
    if dataset_dir.exists():
        return dataset_dir

    root.mkdir(parents=True, exist_ok=True)
    raw = urlopen(UCI_HAR_URL, timeout=180).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = zf.namelist()
        # New UCI wrapper zip contains an inner "UCI HAR Dataset.zip".
        if any(n.endswith("UCI HAR Dataset.zip") for n in names):
            inner_name = next(n for n in names if n.endswith("UCI HAR Dataset.zip"))
            inner = zf.read(inner_name)
            with zipfile.ZipFile(io.BytesIO(inner)) as inner_zf:
                inner_zf.extractall(root)
        else:
            zf.extractall(root)

    if not dataset_dir.exists():
        # Fallback: scan extracted tree for expected folder name.
        for p in root.rglob("UCI HAR Dataset"):
            if p.is_dir():
                return p
        raise FileNotFoundError(f"UCI HAR dataset not found under {root}")
    return dataset_dir


def load_uci_har(root: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset_dir = ensure_uci_har(root)

    channels = [
        "body_acc_x",
        "body_acc_y",
        "body_acc_z",
        "body_gyro_x",
        "body_gyro_y",
        "body_gyro_z",
        "total_acc_x",
        "total_acc_y",
        "total_acc_z",
    ]

    def read_split(split: str) -> tuple[np.ndarray, np.ndarray]:
        sig_dir = dataset_dir / split / "Inertial Signals"
        mats = []
        for ch in channels:
            mats.append(_read_signal_file(sig_dir / f"{ch}_{split}.txt"))
        x = np.stack(mats, axis=-1)  # [N, T, D]
        y = np.loadtxt(dataset_dir / split / f"y_{split}.txt").astype(int) - 1
        return x.astype(np.float32), y.astype(np.int64)

    x_train, y_train = read_split("train")
    x_test, y_test = read_split("test")
    return x_train, y_train, x_test, y_test


def load_npz_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["x_train"], data["y_train"], data["x_test"], data["y_test"]


def fit_channel_standardizer(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x_train.mean(axis=(0, 1), keepdims=True)
    sigma = x_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return mu, sigma


def apply_standardizer(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (x - mu) / sigma
