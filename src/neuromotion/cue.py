"""Cue-log semantics shared across streams (rpi-stim cue_log -> aligned cuelog
-> raw annotations -> per-block / per-trial tables).

Stages: get (config) -> read / parse -> calc / align -> annot -> tag / group.
Supports the Rpi_cue variants (cue_experiment, cue_simple, cue_cadence): all
share one cue_log header and the marker label '{a|v}_{subtype}_p{pos}_t{trial}';
they differ only in block layout (see configs.cue_log_type) and in which
condition columns are filled.

Annotation description schema (written by annot_cues, read by parse_cue_annot):
  cue    : "{modality}/{subtype}/p{pos}/template_index{i}/tempo{+1|-1}/dur_s{s}"
           "/attend{high|low}/focusleg{left|right}/t{trial_num}/cuelog{idx}"
  gostop : "gostop/{go|stop}/t{trial_num}/gostop_pause{s}/cuelog{idx}"
Condition fields are empty except on the trial's regular (REG_POS) row.
"""
import re

import mne
import numpy as np
import pandas as pd


TRIAL_CONDS = ["tempo", "attend", "focusleg", "dur_s"]   # logged on the reg row only

CUE_RE = re.compile(
    r"^(?P<modality>auditory|vibrotactile)/(?P<subtype>\w+)/p(?P<pos>\d+)"
    r"/template_index(?P<template_index>\d*)/tempo(?P<tempo>[+-]?\d*)/dur_s(?P<dur_s>[\d.]*)"
    r"/attend(?P<attend>high|low|)/focusleg(?P<focusleg>left|right|)"
    r"/t(?P<trial_num>\d+)/cuelog(?P<cuelog>\d+)$")
GOSTOP_RE = re.compile(
    r"^gostop/(?P<subtype>go|stop)/t(?P<trial_num>\d+)"
    r"(?:/gostop_pause(?P<gostop_pause_s>[\d.]+))?/cuelog(?P<cuelog>\d+)$")
CUE_PREFIXES = ("auditory/", "vibrotactile/", "gostop")


# ---- get ------------------------------------------------------------------

def get_cue_layout(stim_cfg, cue_log_type):
    """Layout dict for a session's rpi-stim TYPE (default 'cue_experiment'),
    with the session's own BLOCK_DURATION_S overriding the type default."""
    layout = dict(cue_log_type[stim_cfg.get("TYPE", "cue_experiment")])
    layout["BLOCK_DURATION_S"] = stim_cfg.get("BLOCK_DURATION_S", layout["BLOCK_DURATION_S"])
    return layout


# ---- read / parse -----------------------------------------------------------

def read_cuelogs(rpi_dir):
    """All '*_run-{idx}_cuelog-aligned.csv' in rpi_dir as one DataFrame with a
    'cuelog' column (the run idx) and tz-aware 'adjusted_iso'."""
    dfs = []
    for p in sorted(rpi_dir.glob("*_run-*_cuelog-aligned.csv")):
        df = pd.read_csv(p)
        df["adjusted_iso"] = pd.to_datetime(df["adjusted_iso"], utc=True)
        df["cuelog"] = int(re.search(r"_run-(\d+)_", p.name).group(1))
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def read_cue_templates(csv_path, tone_dur_s):
    """templates.csv -> {template_index: [(rel_onset_s, 'low'|'high'), ...]}.
    Each pair row yields a 'low' beat at onset_s and a 'high' beat at
    onset_s + tone_dur_s + intra_gap_s (cue_experiment.py generation order)."""
    df = pd.read_csv(csv_path).sort_values(["template_index", "pair_index"])
    return {int(idx): [b for on, gap in zip(g["onset_s"], g["intra_gap_s"])
                       for b in ((on, "low"), (on + tone_dur_s + gap, "high"))]
            for idx, g in df.groupby("template_index")}


def parse_cue_annot(raw):
    """Cue + gostop annotations of raw -> DataFrame, one row per annotation:
    onset (s since meas_date), duration, kind ('cue'|'gostop') and the parsed
    fields. Trial conditions (TRIAL_CONDS) are copied from the reg row to
    every row of the same (cuelog, trial_num)."""
    a = raw.annotations
    desc = pd.Series(a.description, dtype=str)
    cue, gostop = desc.str.extract(CUE_RE), desc.str.extract(GOSTOP_RE)
    is_cue, is_gostop = cue["cuelog"].notna(), gostop["cuelog"].notna()
    df = pd.concat([cue[is_cue].assign(kind="cue"), gostop[is_gostop].assign(kind="gostop")])
    df = df.mask(df == "")                                  # empty fields -> NaN
    df.insert(0, "onset", a.onset[df.index])
    df.insert(1, "duration", a.duration[df.index])
    df = df.sort_values("onset").reset_index(drop=True)
    for col in ("pos", "template_index", "tempo", "trial_num", "cuelog"):
        df[col] = pd.to_numeric(df[col]).astype("Int64")
    for col in ("dur_s", "gostop_pause_s"):
        df[col] = pd.to_numeric(df[col])
    df[TRIAL_CONDS] = df.groupby(["cuelog", "trial_num"])[TRIAL_CONDS].transform("first")
    return df


# ---- calc / align -----------------------------------------------------------

# ---- annot ------------------------------------------------------------------

def format_cue_desc(r):
    """One aligned-cuelog row -> annotation description (inverse of
    parse_cue_annot; schema in the module docstring)."""
    if r.block_subtype in ("go", "stop"):
        pause = "" if pd.isna(r.gostop_pause_s) else f"/gostop_pause{r.gostop_pause_s:.3f}"
        return f"gostop/{r.block_subtype}/t{int(r.trial_num)}{pause}/cuelog{r.cuelog}"
    fmt = lambda v, f: "" if pd.isna(v) else f(v)
    return (f"{r.modality}/{r.block_subtype}/p{int(r.pos)}"
            f"/template_index{fmt(r.template_index, lambda v: int(v))}"
            f"/tempo{fmt(r.tempo, lambda v: f'{int(v):+d}')}/dur_s{fmt(r.dur_s, lambda v: f'{v:g}')}"
            f"/attend{fmt(r.attend, str)}/focusleg{fmt(r.focusleg, str)}"
            f"/t{int(r.trial_num)}/cuelog{r.cuelog}")


def annot_cues(raw, cuelogs, block_durations):
    """Annotate raw (in place) with every cuelog row inside its iso window:
    cue blocks last block_durations[pos], gostop marks are zero-duration.
    Existing cue/gostop annotations are replaced. Returns raw."""
    meas_date = pd.Timestamp(raw.info["meas_date"])
    start = meas_date + pd.to_timedelta(raw.first_time, unit="s")
    end = start + pd.to_timedelta(raw.times[-1], unit="s")
    rows = cuelogs[cuelogs["adjusted_iso"].between(start, end)]
    cue_annot = mne.Annotations(
        onset=(rows["adjusted_iso"] - meas_date).dt.total_seconds().to_numpy(),
        duration=[0.0 if r.block_subtype in ("go", "stop") else block_durations[int(r.pos)]
                  for r in rows.itertuples()],
        description=[format_cue_desc(r) for r in rows.itertuples()],
        orig_time=raw.info["meas_date"])
    keep = [not d.startswith(CUE_PREFIXES) for d in raw.annotations.description]
    raw.set_annotations(raw.annotations[keep] + cue_annot)
    return raw


# ---- tag / group ------------------------------------------------------------

def tag_cue_blocks(events, cue_df):
    """Tag events (DataFrame with onset / duration, same frame as cue_df) by
    the cue block containing each onset. Returns events with the block's
    parsed fields joined (NaN outside every block) plus block_onset,
    block_duration, within_block (event offset also inside the block) and
    latter_half (onset in the block's second half by time -- the one
    definition of "latter half" used across analyses)."""
    blocks = cue_df[cue_df["kind"] == "cue"].sort_values("onset").reset_index(drop=True)
    onset = events["onset"].to_numpy(dtype=float)
    ends = np.append((blocks["onset"] + blocks["duration"]).to_numpy(), -np.inf)
    j = np.searchsorted(blocks["onset"].to_numpy(), onset, side="right") - 1
    hit = onset < ends[j]                                # j = -1 -> -inf -> no hit
    tags = (blocks.drop(columns=["kind", "gostop_pause_s"])
            .rename(columns={"onset": "block_onset", "duration": "block_duration"})
            .reindex(np.where(hit, j, -1)).reset_index(drop=True))
    tags["within_block"] = hit & (onset + events["duration"].to_numpy() <= ends[j])
    tags["latter_half"] = hit & (onset >= tags["block_onset"] + tags["block_duration"] / 2)
    return pd.concat([events.reset_index(drop=True), tags], axis=1)


def calc_cue_beats(cue_df, templates, onset_delay):
    """Beat onsets of every reg / arr block in cue_df -> DataFrame, one row
    per beat: onset (cue_df's frame), beat ('low'|'high'), half_period_s (reg
    only) plus the block's modality, subtype, pos, template_index, trial_num,
    cuelog. Every beat is shifted by onset_delay[modality]. reg: low/high
    alternating every dur_s / 2 from the block onset until its end. arr:
    templates[template_index] (read_cue_templates)."""
    rows = []
    for b in cue_df[cue_df["subtype"].isin(["reg", "arr"])].itertuples():
        t0 = b.onset + onset_delay[b.modality]
        if b.subtype == "reg":
            half = b.dur_s / 2
            beats = [(t, ("low", "high")[k % 2], half)
                     for k, t in enumerate(np.arange(t0, b.onset + b.duration, half))]
        else:
            beats = [(t0 + rel, beat, np.nan) for rel, beat in templates[int(b.template_index)]]
        rows += [{"onset": t, "beat": beat, "half_period_s": half, "modality": b.modality,
                  "subtype": b.subtype, "pos": b.pos, "template_index": b.template_index,
                  "trial_num": b.trial_num, "cuelog": b.cuelog} for t, beat, half in beats]
    return pd.DataFrame(rows)


def group_cue_trials(cue_df, pos_order):
    """One row per complete (cuelog, trial_num) trial: tmin / tmax spanning
    the blocks in pos_order, plus modality and TRIAL_CONDS. Sorted by tmin."""
    b = cue_df[(cue_df["kind"] == "cue") & cue_df["pos"].isin(pos_order)]
    trials = (b.assign(end=b["onset"] + b["duration"])
               .groupby(["cuelog", "trial_num"])
               .agg(tmin=("onset", "min"), tmax=("end", "max"), n_pos=("pos", "nunique"),
                    modality=("modality", "first"),
                    **{c: (c, "first") for c in TRIAL_CONDS}))
    return (trials[trials["n_pos"] == len(pos_order)].drop(columns="n_pos")
            .reset_index().sort_values("tmin").reset_index(drop=True))
