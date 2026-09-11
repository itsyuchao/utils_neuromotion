from __future__ import annotations
import datetime
import logging
import numpy as np
import warnings
import mne
import matplotlib.pyplot as plt
from pathlib import Path


def save_fig(path: Path, fig=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    (fig or plt.gcf()).savefig(path, bbox_inches="tight", dpi=300)
    logging.info("Saved: %s", path)
    plt.close(fig or plt.gcf())


def assert_iso_synced(*raws, tolerance_s: float = 0.01, labels=None) -> None:
    """Verify that all raws share start wallclock and duration to tolerance_s.

    Use this whenever downstream code (annotation copy, gait/cue cycle
    cropping, event conversion) crosses raw boundaries. The check enforces
    that ``meas_date + first_time`` and total duration agree across all
    inputs, so subsequent code may treat raw-relative seconds as
    interchangeable with wallclock-since-meas_date offsets.

    Parameters
    ----------
    *raws : mne.io.BaseRaw
        Two or more raws expected to be ISO-wallclock aligned (e.g. all
        outputs of a single sync run for the same task).
    tolerance_s : float
        Max allowed mismatch in either start time or total duration.
    labels : list[str] | None
        Human-readable labels for error messages.

    Raises
    ------
    ValueError
        If ``meas_date`` is missing on any raw, or start time / duration
        differ across raws by more than ``tolerance_s``.
    """
    if len(raws) < 2:
        return
    labels = list(labels) if labels else [f"raw{i}" for i in range(len(raws))]
    if len(labels) != len(raws):
        raise ValueError("labels length must match number of raws")

    starts, durs = [], []
    for r, lbl in zip(raws, labels):
        md = r.info["meas_date"]
        if md is None:
            raise ValueError(
                f"{lbl}.info['meas_date'] is None -- cannot verify ISO sync. "
                f"Run the sync step that sets meas_date before chunking across raws."
            )
        starts.append(md + datetime.timedelta(seconds=r.first_time))
        durs.append(r.times[-1])

    s0, d0 = starts[0], durs[0]
    for s, d, lbl in zip(starts[1:], durs[1:], labels[1:]):
        ds = abs((s - s0).total_seconds())
        dd = abs(d - d0)
        if ds > tolerance_s:
            raise ValueError(
                f"start-wallclock mismatch beyond tolerance "
                f"({ds:.4f}s > {tolerance_s}s): "
                f"{labels[0]}={s0.isoformat()} vs {lbl}={s.isoformat()}"
            )
        if dd > tolerance_s:
            raise ValueError(
                f"duration mismatch beyond tolerance "
                f"({dd:.4f}s > {tolerance_s}s): "
                f"{labels[0]}={d0:.3f}s vs {lbl}={d:.3f}s"
            )


def assert_iso_overlap(*raws, labels=None) -> None:
    """Verify that all raws overlap in absolute ISO wallclock time.

    Relaxed companion to :func:`assert_iso_synced`: instead of requiring
    identical start wallclock and duration, this only requires that the
    recordings' wallclock windows ``[meas_date + first_time, meas_date +
    first_time + duration]`` mutually overlap. Use it when segmenting one raw
    by another raw's annotations where the two are separate runs sharing only
    a common time span (e.g. an iEEG run and a motion run recorded in the same
    session). ``meas_date`` is authoritative for cross-raw conversion, so it
    must be set on every input.

    Parameters
    ----------
    *raws : mne.io.BaseRaw
        Two or more raws expected to overlap in ISO wallclock time.
    labels : list[str] | None
        Human-readable labels for error messages.

    Raises
    ------
    ValueError
        If ``meas_date`` is missing on any raw, or the wallclock windows do
        not all mutually overlap.
    """
    if len(raws) < 2:
        return
    labels = list(labels) if labels else [f"raw{i}" for i in range(len(raws))]
    if len(labels) != len(raws):
        raise ValueError("labels length must match number of raws")

    starts, ends = [], []
    for r, lbl in zip(raws, labels):
        md = r.info["meas_date"]
        if md is None:
            raise ValueError(
                f"{lbl}.info['meas_date'] is None -- cannot verify ISO overlap. "
                f"Run the sync step that sets meas_date before chunking across raws."
            )
        start = md + datetime.timedelta(seconds=r.first_time)
        starts.append(start)
        ends.append(start + datetime.timedelta(seconds=r.times[-1]))

    latest_start = max(starts)
    earliest_end = min(ends)
    if latest_start > earliest_end:
        desc = ", ".join(
            f"{lbl}=[{s.isoformat()}, {e.isoformat()}]"
            for lbl, s, e in zip(labels, starts, ends)
        )
        raise ValueError(f"raws do not overlap in ISO wallclock time: {desc}")


def antneuro_ucla_63ch() -> mne.channels.DigMontage:
    """Custom 63-channel antNeuro montage used in the UCLA recordings.

    Derived from MNE's biosemi64: drops the four channels not recorded
    (Fpz, CPz, Iz, P9, P10) and adds the four h-suffixed temporal channels
    (FTT9h, FTT10h, TPP9h, TPP10h) as the midpoint of their named
    neighbors, shifted 1 cm inferior (z - 0.01).
    """
    base = mne.channels.make_standard_montage("biosemi64")
    src = base.get_positions()["ch_pos"]
    ch_pos = {k: v.copy() for k, v in src.items()}
    for ch in ("Fpz", "CPz", "Iz", "P9", "P10"):
        ch_pos.pop(ch, None)

    def _mid_inferior(a, b, drop=0.01):
        p = (np.asarray(src[a]) + np.asarray(src[b])) / 2.0
        p[2] -= drop
        return p

    ch_pos["FTT9h"]  = _mid_inferior("FT7", "T7")
    ch_pos["FTT10h"] = _mid_inferior("FT8", "T8")
    ch_pos["TPP9h"]  = _mid_inferior("TP7", "P7")
    ch_pos["TPP10h"] = _mid_inferior("TP8", "P8")

    return mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")


def _reref_formula(reref_ch):
    """Parse a sequential bipolar iEEG channel name into its source-contact
    formula (pure string math, no data/inst involved).

    The Percept device records three bipolar pairs per hemisphere from a
    4-contact strip (contacts 0-3):
        ZERO_THREE  = V0 - V3
        ONE_THREE   = V1 - V3
        ZERO_TWO    = V0 - V2
    From these, the sequential pairs are derived as:
        ZERO_ONE    = ZERO_THREE - ONE_THREE             (V0 - V1)
        ONE_TWO     = ONE_THREE - ZERO_THREE + ZERO_TWO  (V1 - V2)
        TWO_THREE   = ZERO_THREE - ZERO_TWO              (V2 - V3)

    Parameters
    ----------
    reref_ch : str
        Desired output channel, e.g. ``"ZERO_ONE_LEFT"`` (CONTACT_CONTACT_SIDE).
        Valid contact pairs (order matters): ZERO_ONE, ONE_TWO, TWO_THREE.

    Returns
    -------
    list[(int, str)] | None
        [(coeff, source_ch_name), ...] to sum, or None if reref_ch names a
        reversed pair (e.g. ONE_ZERO) -- order matters for bipolar
        re-referencing, so this warns and is skipped rather than an error.
    """
    contacts = ("ZERO", "ONE", "TWO", "THREE")
    valid_pairs = {"ZERO_ONE", "ONE_TWO", "TWO_THREE"}
    reversed_pairs = {"ONE_ZERO", "TWO_ONE", "THREE_TWO",
                      "THREE_ZERO", "TWO_ZERO", "THREE_ONE"}

    parts = reref_ch.upper().split("_")
    if len(parts) != 3 or parts[0] not in contacts or parts[1] not in contacts:
        raise ValueError(
            f"reref_ch must be in the form CONTACT_CONTACT_SIDE "
            f"(e.g. ZERO_ONE_LEFT), got '{reref_ch}'"
        )
    pair, side = f"{parts[0]}_{parts[1]}", parts[2]

    if pair in reversed_pairs:
        warnings.warn(
            f"Reversed pair '{pair}' requested — order matters for bipolar "
            f"re-referencing. Valid sequential pairs are: {sorted(valid_pairs)}. "
            f"Skipping.",
            UserWarning,
            stacklevel=3,
        )
        return None
    if pair not in valid_pairs:
        raise ValueError(f"Unsupported pair '{pair}'. Valid pairs: {sorted(valid_pairs)}")

    zero_three, one_three, zero_two = (f"ZERO_THREE_{side}", f"ONE_THREE_{side}", f"ZERO_TWO_{side}")
    formulas = {
        "ZERO_ONE":  [(1, zero_three), (-1, one_three)],
        "ONE_TWO":   [(1, one_three), (-1, zero_three), (1, zero_two)],
        "TWO_THREE": [(1, zero_three), (-1, zero_two)],
    }
    return formulas[pair]


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
    to_reref = {ch: _reref_formula(ch) for ch in picks_list if ch not in inst.ch_names}
    to_reref = {ch: formula for ch, formula in to_reref.items() if formula is not None}

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