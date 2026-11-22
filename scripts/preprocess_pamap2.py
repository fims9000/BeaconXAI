#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.client
import json
import re
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

PAMAP2_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip"
ALLOWED_ACTS = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
# 9-axis accel16g from hand/chest/ankle in PAMAP2 row layout.
ACC9_COLS = [4, 5, 6, 21, 22, 23, 38, 39, 40]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess PAMAP2 into project npz format")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--window-length", type=int, default=200)
    p.add_argument("--step", type=int, default=100)
    p.add_argument("--min-purity", type=float, default=0.95)
    p.add_argument("--test-subjects", default="8,9")
    p.add_argument("--out", default="./data/pamap2_acc9_w200s100_p095.npz")
    return p.parse_args()


def _download_pamap2_archive(archive: Path, max_attempts: int = 8) -> Path:
    """Stream the large UCI ZIP to disk and resume after connection drops.

    Partial downloads are never treated as complete. On resume, the HTTP
    server must return 206 with an exact Content-Range; a server that ignores
    Range is handled by restarting from zero. The complete ZIP is CRC checked.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if archive.is_file():
        try:
            with zipfile.ZipFile(archive) as zf:
                if zf.testzip() is None:
                    return archive
        except (OSError, zipfile.BadZipFile):
            pass
        archive.unlink()
    part = archive.with_name(archive.name + ".part")
    last_error = None
    for attempt in range(max_attempts):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "BeaconXAI-reproducible-dataset/1.0",
                   "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urlopen(Request(PAMAP2_URL, headers=headers), timeout=300) as stream:
                code = stream.status
                if code not in (200, 206):
                    raise OSError(f"Unexpected HTTP {code}")
                total = None
                if code == 206:
                    m = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                     stream.headers.get("Content-Range", ""))
                    if not m or int(m.group(1)) != offset:
                        raise OSError("Invalid Content-Range for resumed PAMAP2 ZIP")
                    total = int(m.group(3))
                else:
                    # The server ignored Range; never append an entire 200 response.
                    offset = 0
                    if stream.headers.get("Content-Length"):
                        total = int(stream.headers["Content-Length"])
                with part.open("ab" if offset else "wb") as out:
                    while True:
                        try:
                            chunk = stream.read(4 * 1024 * 1024)
                        except http.client.IncompleteRead as exc:
                            if exc.partial:
                                out.write(exc.partial)
                            raise
                        if not chunk:
                            break
                        out.write(chunk)
                if total is not None and part.stat().st_size != total:
                    raise OSError(f"Partial download: {part.stat().st_size}/{total} bytes")
            with zipfile.ZipFile(part) as zf:
                corrupt = zf.testzip()
                if corrupt is not None:
                    raise zipfile.BadZipFile(f"Corrupt ZIP member: {corrupt}")
            part.replace(archive)
            return archive
        except (OSError, http.client.HTTPException, zipfile.BadZipFile,
                EOFError, ValueError) as exc:
            last_error = exc
            print(f"PAMAP2 download interrupted (attempt {attempt+1}/{max_attempts}, "
                  f"retained {part.stat().st_size if part.exists() else 0} bytes): {exc}",
                  flush=True)
            if attempt + 1 < max_attempts:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Cannot download valid PAMAP2 ZIP after {max_attempts} attempts") from last_error


def _ensure_pamap2(root: Path) -> Path:
    ds = root / "PAMAP2_Dataset"
    if (ds / "Protocol").is_dir() and list((ds / "Protocol").glob("subject10*.dat")):
        return ds
    if ds.exists():
        raise RuntimeError(f"Existing PAMAP2 dataset directory is incomplete: {ds}")
    root.mkdir(parents=True, exist_ok=True)
    archive = _download_pamap2_archive(root / "PAMAP2_Dataset.zip")
    # Extract into a staging directory: interruption cannot leave an
    # apparently usable partial dataset in the destination.
    with tempfile.TemporaryDirectory(prefix="pamap2-extract-", dir=root) as tmp:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp)
        prepared = Path(tmp) / "PAMAP2_Dataset"
        if not (prepared / "Protocol").is_dir() or not list(
                (prepared / "Protocol").glob("subject10*.dat")):
            raise FileNotFoundError("PAMAP2 ZIP lacks expected Protocol subject files")
        prepared.rename(ds)
    archive.unlink(missing_ok=True)
    return ds


def _fill_nans_timewise(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    n, d = out.shape
    t = np.arange(n, dtype=np.float64)
    for j in range(d):
        col = out[:, j]
        nan = np.isnan(col)
        if not np.any(nan):
            continue
        good = ~nan
        if np.sum(good) == 0:
            out[:, j] = 0.0
            continue
        out[:, j] = np.interp(t, t[good], col[good]).astype(np.float32, copy=False)
    return out


def _subject_id_from_name(path: Path) -> int:
    m = re.search(r"subject10([0-9])", path.stem.lower())
    if not m:
        raise ValueError(f"Cannot parse subject id from {path.name}")
    return int(m.group(1))


def _windowize(
    x: np.ndarray,
    y: np.ndarray,
    subject_id: int,
    w: int,
    s: int,
    min_purity: float,
    x_out: list[np.ndarray],
    y_out: list[int],
    sid_out: list[int],
) -> None:
    n = len(y)
    if n < w:
        return
    for st in range(0, n - w + 1, s):
        yy = y[st : st + w]
        vals, cnts = np.unique(yy, return_counts=True)
        k = int(np.argmax(cnts))
        maj = int(vals[k])
        purity = float(cnts[k] / w)
        if purity < min_purity:
            continue
        x_out.append(x[st : st + w].astype(np.float32, copy=False))
        y_out.append(maj)
        sid_out.append(subject_id)


def main() -> None:
    args = parse_args()
    root = Path(args.data_root)
    ds = _ensure_pamap2(root)
    prot = ds / "Protocol"
    files = sorted(prot.glob("subject10*.dat"))
    if not files:
        raise FileNotFoundError(f"No subject10*.dat found in {prot}")

    allowed = np.array(ALLOWED_ACTS, dtype=np.int64)
    test_subjects = {int(x.strip()) for x in args.test_subjects.split(",") if x.strip()}
    label_map = {lab: i for i, lab in enumerate(ALLOWED_ACTS)}

    x_all: list[np.ndarray] = []
    y_all: list[int] = []
    sid_all: list[int] = []

    for f in files:
        sid = _subject_id_from_name(f)
        raw = np.loadtxt(f, dtype=np.float64)
        y = raw[:, 1].astype(np.int64)
        x = raw[:, ACC9_COLS].astype(np.float32)
        keep = np.isin(y, allowed)
        y = y[keep]
        x = x[keep]
        if len(y) == 0:
            continue
        x = _fill_nans_timewise(x)
        _windowize(
            x=x,
            y=y,
            subject_id=sid,
            w=args.window_length,
            s=args.step,
            min_purity=args.min_purity,
            x_out=x_all,
            y_out=y_all,
            sid_out=sid_all,
        )

    if not x_all:
        raise RuntimeError("No windows produced. Lower --min-purity or change window params.")

    X = np.stack(x_all, axis=0).astype(np.float32, copy=False)
    Y_orig = np.array(y_all, dtype=np.int64)
    Y = np.array([label_map[int(v)] for v in Y_orig], dtype=np.int64)
    SID = np.array(sid_all, dtype=np.int64)

    tr = ~np.isin(SID, np.array(sorted(test_subjects), dtype=np.int64))
    te = ~tr
    x_train = X[tr]
    y_train = Y[tr]
    x_test = X[te]
    y_test = Y[te]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        subject_train=SID[tr],
        subject_test=SID[te],
        activity_ids=np.array(ALLOWED_ACTS, dtype=np.int64),
    )
    meta = {
        "path": str(out),
        "window_length": int(args.window_length),
        "step": int(args.step),
        "min_purity": float(args.min_purity),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "n_classes_train": int(len(np.unique(y_train))),
        "n_classes_test": int(len(np.unique(y_test))),
        "test_subjects": sorted(test_subjects),
        "train_subjects": sorted(set(int(v) for v in SID[tr])),
    }
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()

