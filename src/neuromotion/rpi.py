"""Raspberry Pi logs: rpi-fetch (grounding clock: A = electromagnet, B = Motive
record start/stop, M = cue markers relayed from rpi-stim) and rpi-stim cue_logs
(block / ping / gostop rows). Read -> split / parse -> calc -> match / align
the stim log into fetch wallclock (aligned cuelog CSVs) -> QC plots.
Marker labels: '{a|v}_{subtype}_p{pos}_t{trial}' (cue_experiment / cue_simple /
cue_cadence share this format and the cue_log header).
"""
from pathlib import Path
from typing import Iterable, List, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neuromotion.io import fmt_mmss_mmm

RPI_FETCH_COLS = ["channel", "edge", "stamp_ticks", "recv_time_iso", "recv_perf_s", "label"]
MODALITY = {"a": "auditory", "v": "vibrotactile"}
MARKER_RE = r"^(?P<modality>[av])_(?P<subtype>[a-z]+)_p(?P<pos>\d+)_t(?P<trial_num>\d+)$"
FETCH_MARKER_RE = r"^(?P<modality>[av])_(?P<subtype>sil|arr|reg|rate)(?:_p(?P<position>\d+))?_t(?P<trial>\d+)$"

_STIM_NUMERIC_COLS = (
    "trial_num", "recv_perf_s", "template_index",
    "dur_s", "tempo", "sync_rating",
    "rtt1_ms", "rtt2_ms", "rtt3_ms",
)

_MATCH_COLUMNS = [
    "stim_run", "fetch_idx", "marker_label",
    "stim_iso", "stim_perf_s", "fetch_iso", "fetch_perf_s",
]


# ---- read ----------------------------------------------------------------------

def glob_rpi_csv(dirpath: Path, pattern: str = "cue_log_*.csv") -> List[Path]:
    dirpath = Path(dirpath)
    return sorted(dirpath.glob(pattern), key=lambda p: p.name)


def read_rpi_fetch_csv(rpi_path: Union[str, Path]) -> pd.DataFrame:
    """Load an rpi-fetch timestamp log into a DataFrame.

    Accepts the legacy 4/5-column schemas (port, edge, stamp_ticks,
    recv_time_iso[, label]), the previous 6-column schema (..., recv_perf,
    label), or the current 6-column schema that uses ``recv_perf_s`` — a
    local monotonic ``time.perf_counter()`` in seconds captured next to
    ``recv_time_iso`` for every serial edge and UDP marker. Missing columns
    are created empty. Timestamps are parsed to UTC-aware pandas datetimes;
    ``stamp_ticks`` and ``recv_perf_s`` are coerced to float (NaN if
    absent/unparseable).
    """
    raw = pd.read_csv(rpi_path, engine="python", skipinitialspace=True)
    raw.columns = [c.strip().lower() for c in raw.columns]
    # Normalize 'port' -> 'channel' for cross-project uniformity
    if "port" in raw.columns and "channel" not in raw.columns:
        raw = raw.rename(columns={"port": "channel"})
    # Legacy: pre-_s suffix perf column
    if "recv_perf" in raw.columns and "recv_perf_s" not in raw.columns:
        raw = raw.rename(columns={"recv_perf": "recv_perf_s"})
    # Ensure all expected columns exist
    for col in RPI_FETCH_COLS:
        if col not in raw.columns:
            raw[col] = ""
    df = raw[RPI_FETCH_COLS].copy()
    df["channel"] = df["channel"].astype(str).str.strip()
    df["edge"] = df["edge"].astype(str).str.strip().replace({"nan": ""})
    df["label"] = df["label"].astype(str).str.strip().replace({"nan": ""})
    df["recv_time_iso"] = pd.to_datetime(
        df["recv_time_iso"].astype(str).str.strip(), utc=True
    ).dt.tz_convert("UTC")
    df["stamp_ticks"] = pd.to_numeric(df["stamp_ticks"], errors="coerce")
    df["recv_perf_s"] = pd.to_numeric(df["recv_perf_s"], errors="coerce")
    return df


def read_rpi_stim_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load one cue_log CSV (unified block+ping schema, 2026-08+).

    ``recv_time_iso`` is parsed to UTC datetime; ``recv_perf_s`` and every
    RTT column are coerced to float (error tags like 'timeout' become NaN).
    Reading as dtype=str first keeps empty cells as '' so they coerce
    cleanly. Files from older schema versions (attend_high /
    dur_jitter_sign / no focusleg column) raise ValueError.
    """
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]

    required = ("event_type", "recv_time_iso", "recv_perf_s", "marker_label",
                "tempo", "attend", "focusleg")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path}: missing column(s) {missing} — expected the "
            "2026-08+ cue_log schema (tempo / attend / focusleg)"
        )

    df["recv_time_iso"] = pd.to_datetime(
        df["recv_time_iso"], utc=True, errors="coerce"
    )
    for col in _STIM_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---- split / parse -------------------------------------------------------------

def split_rpi_fetch(fetch_df):
    """rpi-fetch log -> (edge_df, udp_df). edge_df: rising edges with
    duration_{ch} columns. udp_df: M-channel rows relayed from rpi-stim, plus
    kind ('cue' | 'other' | None for empty label), modality ('a'|'v'), subtype,
    position, trial parsed from the label."""
    edge_df = calc_edge_durations(fetch_df)
    udp_df = fetch_df[fetch_df["channel"] == "M"].reset_index(drop=True)
    parsed = udp_df["label"].str.extract(FETCH_MARKER_RE)
    parsed.insert(0, "kind", np.where(parsed["modality"].notna(), "cue",
                                      np.where(udp_df["label"] != "", "other", None)))
    parsed[["position", "trial"]] = parsed[["position", "trial"]].apply(pd.to_numeric)
    return edge_df, pd.concat([udp_df, parsed], axis=1)


def split_rpi_stim(df: pd.DataFrame):
    """Split a stim cue_log DataFrame into block, ping, and gostop rows.

    The per-trial condition columns (tempo / attend / focusleg) ride along
    unchanged — they are carried in the CSV itself (populated on each
    trial's regular block row, '' elsewhere), so nothing is stamped here.

    Returns
    -------
    stim_block_df : DataFrame of event_type='block' rows.
    stim_ping_df  : DataFrame of event_type='ping' rows.
    stim_gostop_df : DataFrame of event_type='gostop' rows.
    """
    stim_block_df  = df[df["event_type"] == "block"].reset_index(drop=True)
    stim_ping_df   = df[df["event_type"] == "ping"].reset_index(drop=True)
    stim_gostop_df = df[df["event_type"] == "gostop"].reset_index(drop=True)
    return stim_block_df, stim_ping_df, stim_gostop_df


def parse_rpi_marker(labels):
    """Marker labels ('a_reg_p3_t4', 'v_go_p0_t1', ...) -> DataFrame with
    modality (full name), subtype, pos, trial_num; index follows labels."""
    df = labels.str.extract(MARKER_RE)
    df["modality"] = df["modality"].map(MODALITY)
    return df.astype({"pos": int, "trial_num": int})


def summarize_rpi_stim(stim_df):
    """Block-row summary of one cue_log: n_blocks, t_start / t_end /
    duration_s, and block counts per (modality, subtype) from marker_label."""
    block = stim_df[stim_df["event_type"] == "block"]
    t = block["recv_time_iso"]
    m = block["marker_label"].str.extract(MARKER_RE)
    return {"n_blocks": len(block), "t_start": t.min(), "t_end": t.max(),
            "duration_s": (t.max() - t.min()).total_seconds(),
            "counts": m.groupby(["modality", "subtype"]).size().to_dict()}


# ---- calc ----------------------------------------------------------------------

def calc_edge_durations(df: pd.DataFrame) -> pd.DataFrame:
    """Filter an rpi-fetch DataFrame to rising edges and append per-channel
    duration_{channel} columns (seconds between consecutive + events)."""
    edge_df = df[df["edge"] == "+"].copy()
    channels = np.unique(edge_df["channel"].to_numpy())
    for ch in channels:
        col = f"duration_{ch}"
        edge_df[col] = np.nan
        m = edge_df["channel"].to_numpy() == ch
        edge_df.loc[m, col] = (
            edge_df.loc[m, "recv_time_iso"].diff().dt.total_seconds().to_numpy()
        )
    return edge_df.reset_index(drop=True)


def find_triplet_onsets(t, offsets_s, tol_s):
    """Bool mask over pulse times t (s): True where t0 is the leading pulse of
    a real electromagnet train, i.e. companion pulses exist at t0 + offsets_s[1:]
    within tol_s (configs.emagnet_train). Works on any one monotonic clock
    (rpi recv_perf_s, eeg annotation onsets)."""
    t = np.asarray(t, dtype=float)
    return np.array([all(np.any(np.abs(t - (t0 + off)) <= tol_s) for off in offsets_s[1:])
                     for t0 in t], dtype=bool)


def calc_ping_latency(stim_df):
    """One-way stim->fetch latency (s): mean rtt3 over the cue_log's ping rows
    (not halved: the UDP wifi leg dominates). 0 when rtt3 was not logged."""
    ping = stim_df[stim_df["event_type"] == "ping"]
    return float(ping["rtt3_ms"].mean()) / 1000.0 if "rtt3_ms" in ping else 0.0


def calc_match_delta(stim_match_df: pd.DataFrame) -> pd.DataFrame:
    """Within-device elapsed times + cross-device elapsed-time mismatches.

    For each pair of consecutive successfully-matched markers, this asks: did
    the two Pis agree on how much time elapsed between the two markers? If
    the perf_counters tick at the same rate, ``stim_perf_elapsed`` and
    ``fetch_perf_elapsed`` should match; if NTP keeps the wall clocks in step
    they should also match on the iso side. The cross-device columns
    (``stim - fetch``) plot the residual.

    Filters ``stim_match_df`` to rows where all four time fields
    (``stim_perf_s``, ``stim_iso``, ``fetch_perf_s``, ``fetch_iso``) are
    non-NA, sorts by ``stim_perf_s``, then takes ``np.diff`` per device per
    clock.

    All time values are kept in seconds throughout — same unit as the source
    perf_counter / dt.total_seconds() — so callers can apply a single ms
    conversion at plot time.

    Returns a DataFrame whose first column is the joined-pair label and
    whose remaining columns are (in order):
        marker_pair             "<prev_label> -> <curr_label>" (NaN on row 0)
        stim_perf_elapsed_s     diff of stim_perf_s   (seconds)
        stim_iso_elapsed_s      diff of stim_iso      (seconds)
        fetch_perf_elapsed_s    diff of fetch_perf_s  (seconds)
        fetch_iso_elapsed_s     diff of fetch_iso     (seconds)
        stim-fetch_perf         (stim_perf_elapsed - fetch_perf_elapsed), s
        stim-fetch_iso          (stim_iso_elapsed  - fetch_iso_elapsed),  s

    Row 0 carries NaN diff values (no preceding match to diff against).
    """
    columns = [
        "marker_pair",
        "stim_perf_elapsed_s", "stim_iso_elapsed_s",
        "fetch_perf_elapsed_s", "fetch_iso_elapsed_s",
        "stim-fetch_perf", "stim-fetch_iso",
    ]
    if stim_match_df.empty:
        return pd.DataFrame(columns=columns)

    valid_cols = ["stim_perf_s", "stim_iso", "fetch_perf_s", "fetch_iso"]
    valid = stim_match_df.dropna(subset=valid_cols).copy()
    valid = valid.sort_values("stim_perf_s").reset_index(drop=True)
    if valid.empty:
        return pd.DataFrame(columns=columns)

    stim_perf_d  = valid["stim_perf_s"].diff().round(6) # round to microseconds to avoid spurious float noise
    fetch_perf_d = valid["fetch_perf_s"].diff().round(6)
    stim_iso_d   = valid["stim_iso"].diff().dt.total_seconds()
    fetch_iso_d  = valid["fetch_iso"].diff().dt.total_seconds()

    pair_label = valid["marker_label"].shift(1) + " -> " + valid["marker_label"]

    out = pd.DataFrame({
        "marker_pair": pair_label,
        "stim_perf_elapsed_s":  stim_perf_d,
        "stim_iso_elapsed_s":   stim_iso_d,
        "fetch_perf_elapsed_s": fetch_perf_d,
        "fetch_iso_elapsed_s":  fetch_iso_d,
    })
    out["stim-fetch_perf"] = (out["stim_perf_elapsed_s"] - out["fetch_perf_elapsed_s"]).round(6)
    out["stim-fetch_iso"]  = (out["stim_iso_elapsed_s"]  - out["fetch_iso_elapsed_s"]).round(6)
    return out[columns]


def calc_perf_wallclock(
    stim_match_df: pd.DataFrame, device = "stim"
) -> pd.Series:
    """Reconstruct a wall clock using per_counter_raw() and first iso timestamp as anchor.

    Returns a pandas Series (UTC datetimes) aligned to ``stim_match_df.index``.
    """
    if stim_match_df.empty:
        return pd.Series(pd.NaT, index=stim_match_df.index, dtype="datetime64[ns, UTC]")
    anchor = stim_match_df.iloc[0]
    t0_iso    = anchor[f"{device}_iso"]
    t0_perf_s = anchor[f"{device}_perf_s"]
    return t0_iso + pd.to_timedelta(stim_match_df[f"{device}_perf_s"] - t0_perf_s, unit="s")


def calc_stim_jitter(labels, match_df, delta_df):
    """Per-label stim-side scheduler jitter (s, <= 0): the negative part of
    the matched marker's stim-fetch elapsed-iso delta. Local, not cumulative;
    labels without a fetch match get 0."""
    jitter = delta_df["stim-fetch_iso"].clip(upper=0).fillna(0).to_numpy()
    return labels.map(dict(zip(match_df["marker_label"], jitter))).fillna(0.0)


# ---- match / align -------------------------------------------------------------

def match_rpi_markers(
    stim_block_df: pd.DataFrame,
    fetch_udp_df: pd.DataFrame,
    run_idx: int = 0,
    fetch_excluded_idx: Iterable[int] = None,
    tolerance: float = 0.01,
) -> pd.DataFrame:
    """Iteratively align one stim cue_log's block rows to fetch cue rows.

    A plain inner-join on ``marker_label`` is wrong when one fetch log
    spans multiple stim runs — trial labels (``a_sil_p0_t1`` …) restart at
    each new rpi-stim run, so labels alone repeat across runs. The fetch
    perf_counter, however, never resets. We use labels as the primary key
    and the perf_counter as a tie-breaker for short gaps:

    1. Every fetch cue row whose label equals
       ``stim_block_df.iloc[0]['marker_label']`` is a candidate starting
       alignment for this run.
    2. For each candidate, walk both cursors forward. Advance while labels
       agree. On a mismatch, accept a 1-block skip on either side iff the
       perf_counter elapsed time across the skip agrees within
       ``tolerance`` seconds. Stop at the first unrecoverable mismatch.
    3. Keep the longest run (ties → earliest fetch position).

    For sessions with multiple stim files, pass ``fetch_excluded_idx`` —
    the set of fetch_udp_df row indices already consumed by earlier runs.
    The returned ``fetch_idx`` column is the original fetch_udp_df index
    of each matched row, so callers accumulate exclusions across calls.
    Stim files must be passed in chronological order.

    Output columns:
        stim_run, fetch_idx, marker_label,
        stim_iso, stim_perf_s, fetch_iso, fetch_perf_s
    """
    block = (stim_block_df[["marker_label", "recv_time_iso", "recv_perf_s"]]
             .rename(columns={"recv_time_iso": "stim_iso",
                              "recv_perf_s":   "stim_perf_s"})
             .reset_index(drop=True))

    cue = fetch_udp_df[fetch_udp_df["kind"] == "cue"].copy()
    cue["fetch_idx"] = cue.index.astype(int)
    excluded = {int(k) for k in (fetch_excluded_idx or [])}
    if excluded:
        cue = cue[~cue["fetch_idx"].isin(excluded)]
    cue = (cue[["fetch_idx", "label", "recv_time_iso", "recv_perf_s"]]
           .rename(columns={"label":         "marker_label",
                            "recv_time_iso": "fetch_iso",
                            "recv_perf_s":   "fetch_perf_s"})
           .reset_index(drop=True))

    empty = pd.DataFrame(columns=_MATCH_COLUMNS)
    if block.empty or cue.empty:
        return empty
    first_label = block.iloc[0]["marker_label"]
    if not isinstance(first_label, str) or not first_label.strip():
        return empty

    stim_labels  = block["marker_label"].to_numpy()
    stim_perfs   = block["stim_perf_s"].to_numpy(dtype=float)
    fetch_labels = cue["marker_label"].to_numpy()
    fetch_perfs  = cue["fetch_perf_s"].to_numpy(dtype=float)

    candidates = np.where(fetch_labels == first_label)[0]
    if candidates.size == 0:
        return empty

    # Walk both cursors from each candidate start; on a label mismatch accept
    # one missing entry on either side iff the perf_counter elapsed time
    # across the skip agrees within tolerance; stop at the first
    # unrecoverable mismatch. Keep the longest run.
    best_pairs: list = []
    for start in candidates:
        start = int(start)
        pairs = [(0, start)]
        last_s = float(stim_perfs[0])
        last_f = float(fetch_perfs[start])
        j, i = 1, start + 1
        n_s, n_f = len(stim_labels), len(fetch_labels)
        while j < n_s and i < n_f:
            sl, fl = stim_labels[j], fetch_labels[i]
            sp, fp = float(stim_perfs[j]), float(fetch_perfs[i])
            if sl == fl:
                pairs.append((j, i))
                last_s, last_f = sp, fp
                j += 1; i += 1
                continue
            # 1-block gap, fetch missed a packet: stim[j+1] aligns to fetch[i]
            if j + 1 < n_s and stim_labels[j+1] == fl:
                ns = float(stim_perfs[j+1])
                if abs((ns - last_s) - (fp - last_f)) <= tolerance:
                    pairs.append((j+1, i))
                    last_s, last_f = ns, fp
                    j += 2; i += 1
                    continue
            # 1-block gap, stim missed a block: stim[j] aligns to fetch[i+1]
            if i + 1 < n_f and sl == fetch_labels[i+1]:
                nf = float(fetch_perfs[i+1])
                if abs((sp - last_s) - (nf - last_f)) <= tolerance:
                    pairs.append((j, i+1))
                    last_s, last_f = sp, nf
                    j += 1; i += 2
                    continue
            break
        if len(pairs) > len(best_pairs):
            best_pairs = pairs
    if not best_pairs:
        return empty

    j_idx = [j for j, _ in best_pairs]
    i_idx = [i for _, i in best_pairs]
    merged = pd.concat([
        block.iloc[j_idx].reset_index(drop=True),
        cue.iloc[i_idx][["fetch_idx", "fetch_iso", "fetch_perf_s"]].reset_index(drop=True),
    ], axis=1)
    merged["stim_run"] = run_idx
    return merged[_MATCH_COLUMNS].sort_values("stim_perf_s").reset_index(drop=True)


def align_stim_iso(stim_iso, match_df, one_way_lat_s, jitter_s=0.0):
    """Stim wallclock -> fetch wallclock: anchor the first matched marker at
    its fetch_iso minus one-way latency, extrapolate by stim elapsed iso,
    subtract local jitter."""
    anchor = match_df["fetch_iso"].iloc[0] - pd.to_timedelta(one_way_lat_s, unit="s")
    return anchor + (stim_iso - match_df["stim_iso"].iloc[0]) - pd.to_timedelta(jitter_s, unit="s")


def align_cuelog(stim_df, match_df, delta_df, one_way_lat_s):
    """One stim cue_log -> aligned cuelog DataFrame (block + gostop rows,
    sorted by adjusted_iso in fetch wallclock). Block rows get jitter
    correction; gostop rows (never sent to fetch) only the anchor shift."""
    block = stim_df[stim_df["event_type"] == "block"]
    gostop = stim_df[stim_df["event_type"] == "gostop"]
    block_iso = align_stim_iso(block["recv_time_iso"], match_df, one_way_lat_s,
                              calc_stim_jitter(block["marker_label"], match_df, delta_df))
    gostop_iso = align_stim_iso(gostop["recv_time_iso"], match_df, one_way_lat_s)

    def _rows(df, iso, cue_cols):
        m = parse_rpi_marker(df["marker_label"])
        out = pd.DataFrame({
            "marker_label":  df["marker_label"],
            "trial_num":     df["trial_num"],
            "modality":      m["modality"],
            "pos":           m["pos"],
            "block_subtype": m["subtype"],
        })
        for col in ("tempo", "focusleg", "attend", "dur_s", "template_index", "sync_rating"):
            out[col] = df[col].replace("", np.nan) if cue_cols else np.nan
        out["gostop_pause_s"] = np.nan if cue_cols else pd.to_numeric(df["gostop_pause_s"], errors="coerce")
        out["adjusted_iso"] = iso
        return out

    aligned = (pd.concat([_rows(block, block_iso, True), _rows(gostop, gostop_iso, False)],
                         ignore_index=True)
               .sort_values(by="adjusted_iso").reset_index(drop=True))
    for col in ("trial_num", "tempo", "template_index", "sync_rating"):
        aligned[col] = pd.to_numeric(aligned[col], errors="coerce").astype("Int32")
    return aligned


# ---- plot ----------------------------------------------------------------------

def plot_rpi_fetch_edges(edge_df: pd.DataFrame, min_dur: float = 1.0, ax=None):
    """Stem plot rising edges per channel, annotating durations >= min_dur."""
    channels = np.unique(edge_df["channel"].to_numpy())
    colors = [cm.Set1(i) for i in range(len(channels))]

    if ax is None:
        fig, ax = plt.subplots(
            nrows=1, ncols=1,
            figsize=(10, 6), sharex=False,
        )
    else:
        fig = ax.get_figure()

    for ch_idx, (ch, c) in enumerate(zip(channels, colors)):
        m = edge_df["channel"] == ch
        times = edge_df.loc[m, "recv_time_iso"]
        # Offset y values by channel index for visibility
        y_vals = np.ones(len(times)) * (ch_idx + 1)
        markerline, stemlines, baseline = ax.stem(times, y_vals)
        markerline.set_color(c)
        markerline.set_label(f"Channel {ch} (n={len(times)})")
        stemlines.set_color(c)

        dur_col = f"duration_{ch}"
        if dur_col in edge_df.columns:
            for i, (_, row) in enumerate(edge_df.loc[m].iterrows()):
                dur_val = row[dur_col]
                if pd.notna(dur_val) and dur_val >= min_dur:
                    ax.annotate(
                        fmt_mmss_mmm(dur_val),
                        xy=(row["recv_time_iso"], ch_idx + 1),
                        xytext=(0, (i % 4) * (-8) - 10),  # stagger text vertically
                        textcoords="offset points",
                        ha="center", va="bottom",
                        rotation=45, fontsize=6,
                        color=c
                    )

    ax.set_yticks(np.arange(1, len(channels) + 1))
    ax.set_yticklabels([f"Ch {ch}" for ch in channels])
    ax.set_ylabel("Channel")
    ax.set_xlabel("recv_time_iso")
    ax.legend(loc="upper right")

    fig.autofmt_xdate()
    return fig, ax


def plot_rpi_fetch_markers(fetch_udp_df: pd.DataFrame, ax=None):
    """Rug plot of M cue markers, colored by modality+subtype."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    else:
        fig = ax.get_figure()

    if "kind" in fetch_udp_df.columns:
        cue = fetch_udp_df[fetch_udp_df["kind"] == "cue"].copy()
    elif "modality" in fetch_udp_df.columns:
        cue = fetch_udp_df[fetch_udp_df["modality"].isin(["auditory", "vibrotactile"])].copy()  # in case using synced file
    else:
        cue = fetch_udp_df.iloc[0:0].copy()  # neither column present

    if cue.empty:
        ax.text(0.5, 0.5, "No cue markers parsed",
                transform=ax.transAxes, ha="center")
        return fig, ax

    iso_col = "recv_time_iso" if "recv_time_iso" in cue.columns else "adjusted_iso"
    # Coerce once to tz-aware datetimes: CSV-loaded iso columns arrive as
    # strings, which matplotlib would index categorically and which clash
    # with sibling subplots on a shared datetime x axis.
    cue[iso_col] = pd.to_datetime(cue[iso_col], utc=True)

    subtype_col = cue["subtype"] if "subtype" in cue.columns else cue["block_subtype"]
    cue["key"] = cue["modality"].fillna("?") + "_" + subtype_col.fillna("?")
    keys = sorted(cue["key"].unique())
    colors = cm.tab10(np.linspace(0, 1, max(len(keys), 2)))
    cmap = dict(zip(keys, colors))
    for key in keys:
        sub = cue[cue["key"] == key]
        ax.vlines(sub[iso_col], 0, 1,
                  color=cmap[key], label=f"{key} (n={len(sub)})", alpha=0.8)
    ax.set_yticks([])
    ax.set_xlabel(iso_col)
    ax.set_title(f"M cue markers (n={len(cue)})")
    ax.legend(fontsize=8, loc="upper right")
    fig.autofmt_xdate()
    return fig, ax
