#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import mne
import numpy as np
from mne.datasets.sleep_physionet.age import fetch_data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess Sleep-EDF into project NPZ format")
    p.add_argument("--data-root", default="./data/sleep_edf_raw")
    p.add_argument("--local-dir", default="", help="Use local directory with *-PSG.edf/*-Hypnogram.edf; skip network fetch")
    p.add_argument("--subjects", default="0-19", help="e.g. 0-19 or 0,1,2,5")
    p.add_argument("--recordings", default="1", help="1 or 1,2")
    p.add_argument("--target-sfreq", type=float, default=100.0)
    p.add_argument("--epoch-sec", type=float, default=30.0)
    p.add_argument("--test-subjects", default="", help="comma-separated subject ids; empty -> last 20%%")
    p.add_argument("--max-recordings", type=int, default=0, help="0 means all")
    p.add_argument("--out", default="./data/sleep_edf_4ch_e30_s100_bin.npz")
    return p.parse_args()


def _collect_local_pairs(local_dir: Path) -> list[tuple[str, str]]:
    psg_files = sorted(local_dir.glob("*-PSG.edf"))
    out: list[tuple[str, str]] = []
    for psg in psg_files:
        hyp = local_dir / psg.name.replace("-PSG.edf", "-Hypnogram.edf")
        if hyp.exists():
            out.append((str(psg), str(hyp)))
    return out


def _subject_from_psg_name(psg_path: str) -> int | None:
    stem = Path(psg_path).stem
    m = re.search(r"SC(\d{4})", stem)
    if not m:
        return None
    code = m.group(1)
    return int(code[1:3])


def _parse_int_set(spec: str) -> list[int]:
    spec = spec.strip()
    if not spec:
        return []
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    vals: list[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            lo, hi = tok.split("-", 1)
            vals.extend(range(int(lo), int(hi) + 1))
        else:
            vals.append(int(tok))
    return sorted(set(vals))


def _pick_4ch(raw: mne.io.BaseRaw) -> list[str] | None:
    ch_low = [c.lower() for c in raw.ch_names]
    needed = [
        ["eeg fpz-cz"],
        ["eeg pz-oz"],
        ["eog horizontal", "eog hor"],
        ["emg submental", "emg submentalis", "emg"],
    ]
    picked: list[str] = []
    for aliases in needed:
        found = None
        for i, c in enumerate(ch_low):
            if any(a in c for a in aliases):
                found = raw.ch_names[i]
                break
        if found is None:
            return None
        picked.append(found)
    return picked


def _extract_epochs_binary(
    psg_path: str,
    hyp_path: str,
    target_sfreq: float,
    epoch_sec: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose="ERROR")
    pick = _pick_4ch(raw)
    if pick is None:
        return None
    raw.pick(pick)
    raw.resample(target_sfreq, npad="auto")

    ann = mne.read_annotations(hyp_path)
    raw.set_annotations(ann, emit_warning=False)
    event_id = {
        "Sleep stage W": 1,
        "Sleep stage 1": 2,
        "Sleep stage 2": 2,
        "Sleep stage 3": 2,
        "Sleep stage 4": 2,
        "Sleep stage R": 2,
    }
    events, _ = mne.events_from_annotations(
        raw,
        event_id=event_id,
        chunk_duration=float(epoch_sec),
        verbose="ERROR",
    )
    if len(events) == 0:
        return None
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=0.0,
        tmax=float(epoch_sec) - 1.0 / float(target_sfreq),
        baseline=None,
        preload=True,
        verbose="ERROR",
    )
    x = epochs.get_data(copy=False).astype(np.float32, copy=False)  # [N,C,T]
    y = epochs.events[:, 2].astype(np.int64)  # 1=wake, 2=sleep
    y = (y - 1).astype(np.int64, copy=False)  # 0/1
    x = np.transpose(x, (0, 2, 1))  # [N,T,C]
    return x, y


def main() -> None:
    args = parse_args()
    subjects = _parse_int_set(args.subjects)
    recordings = _parse_int_set(args.recordings)
    if not subjects:
        raise ValueError("No subjects parsed from --subjects")
    if not recordings:
        recordings = [1]

    pairs: list[tuple[str, str]]
    if args.local_dir.strip():
        pairs = _collect_local_pairs(Path(args.local_dir))
        if not pairs:
            raise RuntimeError(f"No local PSG/Hypnogram pairs found in {args.local_dir}")
        # keep only selected subject ids
        filt = []
        allow = set(subjects)
        for psg, hyp in pairs:
            sid = _subject_from_psg_name(psg)
            if sid is None:
                continue
            if sid in allow:
                filt.append((psg, hyp))
        pairs = filt
        if not pairs:
            raise RuntimeError("No local pairs left after subject filtering")
    else:
        data_root = Path(args.data_root)
        data_root.mkdir(parents=True, exist_ok=True)
        pairs = fetch_data(
            subjects=subjects,
            recording=recordings,
            path=str(data_root),
            on_missing="warn",
        )
    if args.max_recordings > 0:
        pairs = pairs[: args.max_recordings]
    if not pairs:
        raise RuntimeError("No Sleep-EDF files fetched")

    x_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    sid_all: list[np.ndarray] = []

    for psg, hyp in pairs:
        sid = _subject_from_psg_name(psg)
        if sid is None:
            continue
        out = _extract_epochs_binary(
            psg_path=psg,
            hyp_path=hyp,
            target_sfreq=args.target_sfreq,
            epoch_sec=args.epoch_sec,
        )
        if out is None:
            continue
        x, y = out
        if len(y) == 0:
            continue
        x_all.append(x)
        y_all.append(y)
        sid_all.append(np.full(len(y), sid, dtype=np.int64))

    if not x_all:
        raise RuntimeError("No epochs extracted. Check channels/subjects.")

    X = np.concatenate(x_all, axis=0).astype(np.float32, copy=False)
    Y = np.concatenate(y_all, axis=0).astype(np.int64, copy=False)
    SID = np.concatenate(sid_all, axis=0).astype(np.int64, copy=False)

    uniq_sid = sorted(set(int(v) for v in SID))
    if args.test_subjects.strip():
        test_subjects = set(_parse_int_set(args.test_subjects))
    else:
        n_te = max(1, int(round(0.2 * len(uniq_sid))))
        test_subjects = set(uniq_sid[-n_te:])
    tr = ~np.isin(SID, np.array(sorted(test_subjects), dtype=np.int64))
    te = ~tr

    x_train, y_train = X[tr], Y[tr]
    x_test, y_test = X[te], Y[te]
    sid_tr, sid_te = SID[tr], SID[te]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        subject_train=sid_tr,
        subject_test=sid_te,
        class_names=np.array(["wake", "sleep"], dtype=object),
        channel_names=np.array(["EEG_Fpz-Cz", "EEG_Pz-Oz", "EOG", "EMG"], dtype=object),
    )

    meta = {
        "path": str(out),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "shape": [int(x_train.shape[1]), int(x_train.shape[2])],
        "n_subjects_total": int(len(uniq_sid)),
        "train_subjects": sorted(set(int(v) for v in sid_tr)),
        "test_subjects": sorted(set(int(v) for v in sid_te)),
        "class_balance_train": {
            "wake": int(np.sum(y_train == 0)),
            "sleep": int(np.sum(y_train == 1)),
        },
        "class_balance_test": {
            "wake": int(np.sum(y_test == 0)),
            "sleep": int(np.sum(y_test == 1)),
        },
    }
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
