"""Medtronic Percept JSON reports (IndefiniteStreaming / BrainSenseTimeDomain).

read  -> read_percept_json (raw dict)
parse -> summarize_percept_runs (one row per run), parse_percept_run (one run
         as an MNE RawArray), parse_percept_lfp (BrainSenseLfp stim mA)
plot  -> plot_percept_packets, plot_percept_runs, plot_magnet_traces
reref -> pick_or_reref (sequential bipolar pairs from Percept contact pairs)

A report's streaming "session" holds continuous segments, called runs (BIDS);
run r occupies entries [r * n, (r + 1) * n) of data[stream_type], with n
channel entries per run fixed by the stream type (N_CH_PER_RUN).
"""
import json

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from neuromotion.io import fmt_mmss_mmm

N_CH_PER_RUN = {"IndefiniteStreaming": 6, "BrainSenseTimeDomain": 1}

# Sequential bipolar pairs from the three recorded pairs of a 4-contact lead
# (ZERO_THREE = V0-V3, ONE_THREE = V1-V3, ZERO_TWO = V0-V2), per side:
#   ZERO_ONE = V0-V1, ONE_TWO = V1-V2, TWO_THREE = V2-V3
BIPOLAR_FORMULA = {
    "ZERO_ONE":  [(1, "ZERO_THREE"), (-1, "ONE_THREE")],
    "ONE_TWO":   [(1, "ONE_THREE"), (-1, "ZERO_THREE"), (1, "ZERO_TWO")],
    "TWO_THREE": [(1, "ZERO_THREE"), (-1, "ZERO_TWO")],
}


# ---- read -------------------------------------------------------------------

def read_percept_json(json_path):
    """Percept JSON report -> dict."""
    with open(json_path) as f:
        return json.load(f)


# ---- parse ------------------------------------------------------------------

def get_percept_entries(data, run, stream_type):
    """data[stream_type] entries (one per channel) of one run."""
    n = N_CH_PER_RUN[stream_type]
    return data[stream_type][run * n:(run + 1) * n]


def summarize_percept_runs(data, stream_type="IndefiniteStreaming"):
    """One row per run: start_s (first tick, session time), duration_s
    (samples / sfreq), tick_duration_s (ticks are only 0.25 s precise),
    n_samples, sfreq, n_missing_packets (GlobalSequences gaps) and
    packets_match (GlobalPacketSizes sum == n_samples)."""
    rows = []
    for run in range(len(data[stream_type]) // N_CH_PER_RUN[stream_type]):
        e = get_percept_entries(data, run, stream_type)[0]
        ints = lambda key: np.array(e[key].strip(",").split(","), dtype=int)   # "1,2,3," -> [1, 2, 3]
        ticks, seq = ints("TicksInMses") / 1000.0, ints("GlobalSequences")
        n, fs = len(e["TimeDomainData"]), e["SampleRateInHz"]
        rows.append({"run": run, "start_s": ticks[0], "duration_s": n / fs,
                     "tick_duration_s": ticks[-1] - ticks[0], "n_samples": n, "sfreq": fs,
                     "n_missing_packets": int(np.sum(np.diff(seq) != 1)),
                     "packets_match": ints("GlobalPacketSizes").sum() == n})
    df = pd.DataFrame(rows)
    print(f"{data['SessionDate']}: AbnormalEnd={data['AbnormalEnd']}, {len(df)} {stream_type} run(s)")
    return df


def parse_percept_run(data, run, stream_type="IndefiniteStreaming"):
    """One run -> RawArray (seeg, Percept device units, t=0 at its first sample)."""
    entries = get_percept_entries(data, run, stream_type)
    info = mne.create_info([e["Channel"] for e in entries], sfreq=entries[0]["SampleRateInHz"], ch_types="seeg")
    raw = mne.io.RawArray(np.vstack([e["TimeDomainData"] for e in entries]), info, verbose=False)
    raw.info["description"] = f"Percept:{data['SessionDate']}_run={run}"
    return raw


def parse_percept_lfp(data, run):
    """BrainSenseLfp stim intensity of one BrainSenseTimeDomain run -> (t,
    left_mA, right_mA); t (s) from the TimeDomain run's first tick, built from
    each sample's own TicksInMs (2 or 20 Hz stream, robust to drops)."""
    td_start_ms = np.array(get_percept_entries(data, run, "BrainSenseTimeDomain")[0]["TicksInMses"].strip(",").split(","), dtype=int)[0]
    lfp = data["BrainSenseLfp"][run * N_CH_PER_RUN["BrainSenseTimeDomain"]]["LfpData"]
    t = (np.array([s["TicksInMs"] for s in lfp]) - td_start_ms) / 1000.0
    return t, np.array([s["Left"]["mA"] for s in lfp]), np.array([s["Right"]["mA"] for s in lfp])


# ---- plot -------------------------------------------------------------------

def plot_percept_packets(data, stream_type="IndefiniteStreaming"):
    """GlobalSequences per run; red dashed lines (first 20 labeled) at gaps."""
    n_runs = len(data[stream_type]) // N_CH_PER_RUN[stream_type]
    fig, axes = plt.subplots(1, n_runs, figsize=(4 * n_runs, 4), squeeze=False)
    for run, ax in enumerate(axes[0]):
        seq = np.array(get_percept_entries(data, run, stream_type)[0]["GlobalSequences"].strip(",").split(","), dtype=int)
        ax.plot(seq)
        gaps = np.flatnonzero(np.diff(seq) != 1)
        for j, g in enumerate(gaps):
            ax.axvline(g + 1, color="red", ls="--")
            if j < 20:
                y0, y1 = ax.get_ylim()
                ax.text(ax.get_xlim()[0], y0 + (y1 - y0) / 10 * j, f"Previous: {seq[g]}; Current: {seq[g + 1]}",
                        fontsize=8, color="red")
        ax.set(title=f"Run {run} GlobalSequences", xlabel="Packet Index", ylabel="GlobalSequence")
    return fig


def plot_percept_runs(runs_df, magnets_in_run, title=""):
    """Runs (orange spans, session time) with configured magnet times and the
    interval to the previous magnet."""
    fig, ax = plt.subplots(figsize=(12, 3))
    magnets = []
    for r in runs_df.itertuples():
        end = r.start_s + r.duration_s
        ax.axvspan(r.start_s, end, color="orange", alpha=0.3)
        ax.text((r.start_s + end) / 2, 0, f"Run {r.run} \n Duration {r.duration_s}", color="orange",
                ha="center", va="bottom", fontsize=10)
        magnets += [m + r.start_s for m in magnets_in_run.get(r.run) or []]
    if magnets:
        ax.stem(magnets, np.ones(len(magnets)), linefmt="C0-", markerfmt="C0o", basefmt=" ")
        for i, (prev, m) in enumerate(zip(magnets, magnets[1:]), start=1):
            ax.text(m, (i % 5) * 0.1 + 0.5, f"{fmt_mmss_mmm(m - prev)}s", rotation=45, va="bottom",
                    ha="center", fontsize=8)
    ax.set(xlabel="Time (seconds)", title=title)
    return fig


def plot_magnet_traces(raw, magnet_times, pre_s=0.2, post_s=1.0):
    """All channels around each magnet time (s, raw-relative), one panel per
    magnet on a 4x4 grid."""
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    data, t = raw.get_data(), raw.times
    for i, ax in enumerate(axes.ravel()):
        if i >= len(magnet_times):
            ax.axis("off")
            continue
        sel = (t >= magnet_times[i] - pre_s) & (t < magnet_times[i] + post_s)
        for trace, ch in zip(data, raw.ch_names):
            ax.plot(t[sel], trace[sel], lw=1, alpha=0.7, label=ch if i == 0 else None)
        ax.set_title(f"Magnet pulse {i}")
    fig.suptitle(f"{raw.info['description']}: {len(magnet_times)} magnets")
    fig.legend(*axes.ravel()[0].get_legend_handles_labels(), loc="upper right")
    return fig


# ---- reref ------------------------------------------------------------------

def pick_or_reref(inst: mne.io.BaseRaw | mne.BaseEpochs, ieeg_picks: list[str] | str):
    """Pick channels from inst, re-referencing any that don't exist as-is.

    Works on inst.copy() throughout rather than building a fresh Raw/Epochs
    for the derived channels: every derived channel's data is computed first
    (while every original source channel is still untouched -- sequential
    bipolar formulas reuse source contacts across targets, e.g. ONE_TWO needs
    the same ZERO_THREE/ZERO_TWO contacts as TWO_THREE, so overwriting one
    target's carrier channel before all formulas are evaluated would corrupt
    a still-needed source), then each result is written in place over a
    spare (not requested) source channel, which is renamed to the derived
    channel's name. The returned object is that same copy of inst, so
    meas_date, annotations/events, description, and everything else about
    inst's metadata carry over automatically -- there's nothing to copy by
    hand, and nothing to get out of sync.

    Note: because derived channels are carved out of inst's own spare
    channels rather than added fresh, a call can't request both a raw
    source contact AND a derived channel built from it in the same
    `ieeg_picks` if that leaves too few spare channels to hold every
    derived channel -- this raises ValueError rather than silently
    dropping one.
    """
    picks_list = ieeg_picks if isinstance(ieeg_picks, list) else [ieeg_picks]
    to_reref = {}
    for ch in picks_list:
        if ch not in inst.ch_names:
            pair, side = ch.upper().rsplit("_", 1)
            if pair not in BIPOLAR_FORMULA:
                raise ValueError(f"{ch}: derivable pairs are {list(BIPOLAR_FORMULA)} + _LEFT/_RIGHT (order matters)")
            to_reref[ch] = [(coeff, f"{src}_{side}") for coeff, src in BIPOLAR_FORMULA[pair]]

    out = inst.copy()
    out.load_data()

    # Compute every derived channel's data up front, before any carrier
    # channel is overwritten (see docstring).
    computed = {}
    for ch, formula in to_reref.items():
        needed = [src_ch for _, src_ch in formula]
        missing = [src_ch for src_ch in needed if src_ch not in out.ch_names]
        if missing:
            raise ValueError(f"Source channels {missing} not found in inst.ch_names: {out.ch_names}")
        data = out.get_data(picks=needed)  # (n_channels, n_times) or (n_epochs, n_channels, n_times)
        ch_idx = {src_ch: i for i, src_ch in enumerate(needed)}
        if isinstance(out, mne.BaseEpochs):
            computed[ch] = sum(coeff * data[:, ch_idx[src_ch], :] for coeff, src_ch in formula)
        else:
            computed[ch] = sum(coeff * data[ch_idx[src_ch]] for coeff, src_ch in formula)

    carriers = [ch for ch in out.ch_names if ch not in picks_list]
    if len(carriers) < len(computed):
        raise ValueError(
            f"pick_or_reref repurposes inst's own channels in place and can't add new "
            f"ones: need {len(computed)} spare channel(s) for {list(computed)}, only "
            f"{len(carriers)} available ({carriers})"
        )

    for carrier, (ch, data) in zip(carriers, computed.items()):
        idx = out.ch_names.index(carrier)
        if isinstance(out, mne.BaseEpochs):
            out._data[:, idx, :] = data
        else:
            out._data[idx] = data
        out.rename_channels({carrier: ch})

    if not any(ch in out.ch_names for ch in picks_list):
        raise ValueError(f"No valid channels from {picks_list} in {inst.ch_names}")
    return out.pick([ch for ch in picks_list if ch in out.ch_names])
