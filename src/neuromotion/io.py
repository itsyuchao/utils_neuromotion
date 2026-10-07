from __future__ import annotations
import logging
import mne
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from bids import BIDSLayout


def save_fig(path: Path, fig=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    (fig or plt.gcf()).savefig(path, bbox_inches="tight", dpi=300)
    logging.info("Saved: %s", path)
    plt.close(fig or plt.gcf())


# ---- iso time: every cross-stream window is given in a reference raw's
# annotation frame (s since its meas_date) and moved between raws through
# meas_date only ------------------------------------------------------------

def get_iso_span(raw):
    """(start, end) tz-aware wallclock of raw's first and last sample."""
    start = pd.Timestamp(raw.info["meas_date"]) + pd.to_timedelta(raw.first_time, unit="s")
    return start, start + pd.to_timedelta(raw.times[-1], unit="s")


def iso_to_onset(raw, iso):
    """Wallclock -> raw's annotation frame (s since meas_date)."""
    return (pd.Timestamp(iso) - pd.Timestamp(raw.info["meas_date"])).total_seconds()


def calc_iso_shift(raw_from, raw_to):
    """Seconds to add to raw_from annotation-frame times to express them in
    raw_to's annotation frame."""
    return (raw_from.info["meas_date"] - raw_to.info["meas_date"]).total_seconds()


def find_overlapping(span, paths, cover=False):
    """fif paths whose iso span overlaps span=(start, end); cover=True keeps
    only those fully covering it. Headers only."""
    out = []
    for p in paths:
        s, e = get_iso_span(mne.io.read_raw_fif(p, preload=False, verbose="ERROR"))
        if (s <= span[0] and span[1] <= e) if cover else (s <= span[1] and span[0] <= e):
            out.append(p)
    return out


def assert_iso_synced(*raws, tolerance_s=0.01):
    """Raise if raws differ in start wallclock or duration by > tolerance_s
    (e.g. eeg / ieeg / motion cut from one sync run of the same task)."""
    spans = [get_iso_span(r) for r in raws]
    for (s, e), r in zip(spans[1:], raws[1:]):
        ds = abs((s - spans[0][0]).total_seconds())
        dd = abs((e - s).total_seconds() - (spans[0][1] - spans[0][0]).total_seconds())
        if ds > tolerance_s or dd > tolerance_s:
            raise ValueError(f"{r.info['description']}: start off by {ds:.4f}s, duration by {dd:.4f}s")


def split_windows(periods, win_s):
    """Split each (t0, t1) period into back-to-back win_s windows (remainder
    dropped) -> DataFrame onset, duration, period_idx; input for crop_windows."""
    return pd.DataFrame([(t0 + k * win_s, win_s, i) for i, (t0, t1) in enumerate(periods)
                         for k in range(int((t1 - t0) // win_s))],
                        columns=["onset", "duration", "period_idx"])


def crop_windows(raw, windows, pad_s=0.0, raw_ref=None, mid=None, verbose=True):
    """Crop padded segments of raw around each window -- the one cropper.

    Parameters
    ----------
    windows : DataFrame with 'onset' / 'duration' (s, annotation frame of
        raw_ref if given, else of raw). Every column is carried into that
        window's info dict.
    pad_s : buffer before/after each window, kept in the segment so a later
        Morlet/Hilbert transform can trim it via cycle_start_idx/_end_idx.
    mid : optional column (same frame as 'onset') stored as cycle_mid_idx,
        e.g. 'right_onset' for gait cycles.

    Returns
    -------
    epochs : list of Raw segments (pads included); windows whose padded span
        leaves raw are dropped.
    info : list of dict, 1:1 with epochs: the window's columns (onset in raw's
        frame) + sfreq, pad_s, n_samples, cycle_start_idx, cycle_end_idx
        [, cycle_mid_idx].
    """
    sfreq = raw.info["sfreq"]
    shift = 0.0 if raw_ref is None else calc_iso_shift(raw_ref, raw)
    pad_samp = int(round(pad_s * sfreq))
    epochs, info = [], []
    for row in windows.to_dict("records"):
        t0 = row["onset"] + shift - raw.first_time - pad_s
        t1 = t0 + row["duration"] + 2 * pad_s
        if t0 < 0 or t1 > raw.times[-1]:
            continue
        ep = raw.copy().crop(tmin=t0, tmax=t1, include_tmax=False)
        ci = {**row, "onset": row["onset"] + shift, "sfreq": sfreq, "pad_s": pad_s,
              "n_samples": ep.n_times, "cycle_start_idx": pad_samp,
              "cycle_end_idx": ep.n_times - pad_samp}
        if mid is not None:
            ci["cycle_mid_idx"] = pad_samp + int(round((row[mid] - row["onset"]) * sfreq))
        epochs.append(ep)
        info.append(ci)
    if verbose:
        print(f"crop_windows: {len(epochs)}/{len(windows)} window(s) inside raw (pad_s={pad_s})")
    return epochs, info


def get_matched_window(raw_ref, raw, tmin, tmax, picks=None):
    """raw's data over [tmin, tmax] given in raw_ref's annotation frame.
    Returns (t, data): t in raw_ref's annotation frame, data (n_picks, n)."""
    win = pd.DataFrame({"onset": [tmin], "duration": [tmax - tmin]})
    (ep,), _ = crop_windows(raw.copy().pick(picks), win, raw_ref=raw_ref, verbose=False)
    return ep.times + ep.first_time - calc_iso_shift(raw_ref, raw), ep.get_data()


def fmt_mmss_mmm(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds - m*60
    return f"{m:02d}:{s:06.3f}"


def list_saved_runs(derivs_root, subject, task="selflocation", datatype="ieeg", suffix="ieeg", extension=".fif"):
    layout = BIDSLayout(derivs_root, derivatives=True, validate=False)
    files = layout.get(
        subject=subject, task=task, datatype=datatype, suffix=suffix, extension=extension,
        return_type="filename"
    )
    # Extract run numbers from BIDS entities
    runs = sorted({layout.parse_file_entities(f).get("run") for f in files})
    return [r for r in runs if r is not None], files
