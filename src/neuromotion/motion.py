"""OptiTrack Motive motion capture: read takes -> Raw, gait annotations
(lr_step / gait_lean) and gait cycles, kinematics (speed, heading, bones,
joint angles, per-trial matrices) and motion plots (paths, skeletons).
Ground-plane / vertical axes are always passed explicitly (configs.GROUND_AXES,
VERTICAL_AXIS). Positions are meters from parse_motive_raw on (Motive CSVs are
mm); speeds m/s, rotations degrees.
"""
import re
from io import StringIO
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import mne
import numpy as np
import pandas as pd

from neuromotion.io import fmt_mmss_mmm
from neuromotion.calc import interp_vector, calc_moving_average, calc_bin_means

# Smoothing windows (s). Heading: gradient of the path averaged over ~1 gait
# cycle (median cycle 1.0-1.2 s here), so step-to-step sway cancels while turns
# are kept. Speed: body speed averaged over ~1/5 cycle, keeping within-cycle
# speed changes but not sample jitter.
HEADING_SMOOTH_S = 1.0
SPEED_SMOOTH_S = 0.2
STEP_COLOR = {-1: "#e53935", 0: "#222222", 1: "#1e88e5"}   # left swing / reset / right swing


# ---- read ----------------------------------------------------------------------

def glob_motive_csv(
    dirpath: Path,
    suffix: Union[str, None] = None,
) -> pd.DataFrame:
    """Discover Motive CSV exports in `dirpath` and summarize frame counts.

    `suffix=None` matches all `*.csv`; otherwise the glob is `*{suffix}*.csv`,
    which matches both per-asset exports (`...-Handshake.csv`) and full-take
    exports (`Take ....csv`).
    """
    dirpath = Path(dirpath)
    pattern = f"*{suffix}*.csv" if suffix else "*.csv"
    matches = sorted(dirpath.glob(pattern), key=lambda p: p.name)
    motive_df = pd.DataFrame()
    for match in matches:
        header = pd.read_csv(match, nrows=1)
        header_dict = dict(zip(header.columns[0::2], header.columns[1::2]))
        total_frames = int(header_dict["Total Frames in Take"])
        srate = float(header_dict["Capture Frame Rate"])
        dur_s = round(total_frames / srate, 3)
        new_row = pd.DataFrame(
            {"csv_path": [match],
             "TotalFramesinTake": [total_frames],
             "DurationinSeconds": [dur_s]}
        )
        motive_df = pd.concat([motive_df, new_row], ignore_index=True)
        print(f"Found Motive CSV file: {match.name}")
        print(f"Total Frames: {total_frames} with {fmt_mmss_mmm(dur_s)} total duration at {srate} Hz")

    return motive_df


def read_motive_csv(
    csv_path: Union[str, Path],
    rigid_body: Union[str, List[str]] = "Handshake",
    rotation: bool = False,
):
    """Read an OptiTrack/Motive CSV for one or more trackers.

    Parameters
    ----------
    csv_path : str | Path
    rigid_body : str or list[str]
        A single tracker name (backward-compat) returns an ``(n_frames, 3)``
        ndarray, or ``(n_frames, 6)`` with rotation=True. A list returns a
        DataFrame whose columns are ``{tracker}_Position_X``, ``..._Y``,
        ``..._Z`` (plus ``{tracker}_Rotation_*`` if rotation=True), indexed
        by Frame with a 'Time' column preserved.
        Names are matched exactly against the 'Name' header row OR by the
        suffix after ':' — e.g. 'LFoot' matches 'Skeleton 002:LFoot'.
    rotation : bool
        Include Rotation XYZ after Position XYZ (applies to every tracker).

    Notes
    -----
    Robust to column order: uses the axis labels X/Y/Z from the header. CSVs
    with multiple rigid bodies/bones are supported; only the requested
    tracker columns are returned.
    """
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", errors="replace") as f:
        lines = f.read().splitlines()

    def split_csv_line(line: str) -> List[str]:
        return [c.strip() for c in line.split(",")]

    header_idx = None
    for i, line in enumerate(lines):
        cells = split_csv_line(line)
        if len(cells) >= 2 and cells[0] == "Frame" and "Time" in cells[1]:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find the data header row starting with 'Frame,Time ...'.")
    if header_idx < 3:
        raise ValueError("CSV does not contain enough header rows to parse rigid body columns.")

    hdr_name = split_csv_line(lines[header_idx - 3])
    hdr_grp = split_csv_line(lines[header_idx - 1])
    hdr_axis = split_csv_line(lines[header_idx])

    n_cols = max(len(hdr_name), len(hdr_grp), len(hdr_axis))
    def pad(lst): return lst + [""] * (n_cols - len(lst))
    hdr_name, hdr_grp, hdr_axis = pad(hdr_name), pad(hdr_grp), pad(hdr_axis)

    axes_wanted = ("X", "Y", "Z")
    groups_wanted = ("Position", "Rotation") if rotation else ("Position",)

    single = isinstance(rigid_body, str)
    trackers: List[str] = [rigid_body] if single else list(rigid_body)

    # Build {tracker: {"<grp>_<ax>": col_idx}}
    col_for: dict[str, dict[str, int]] = {t: {} for t in trackers}
    for c in range(2, n_cols):
        name, grp, ax = hdr_name[c], hdr_grp[c], hdr_axis[c]
        if grp not in groups_wanted or ax not in axes_wanted:
            continue
        for t in trackers:
            if name == t or name.endswith(f":{t}"):   # rigid body 'Head' or bone 'Skeleton 002:LFoot'
                col_for[t][f"{grp}_{ax}"] = c
                break

    wanted_keys = [f"{g}_{a}" for g in groups_wanted for a in axes_wanted]
    for t in trackers:
        missing = [k for k in wanted_keys if k not in col_for[t]]
        if missing:
            available = sorted(col_for[t].keys())
            raise ValueError(
                f"Could not find required columns for tracker='{t}'. "
                f"Missing: {missing}. Found: {available}. "
                f"Hint: check the exact tracker name in the CSV header 'Name' row."
            )

    data_str = "\n".join(lines[header_idx + 1:])
    df = pd.read_csv(StringIO(data_str), header=None)

    if single:
        cols = [col_for[trackers[0]][k] for k in wanted_keys]
        return df.loc[:, cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    # Multi-tracker: build all columns first, then assemble in one shot
    # (assigning into an existing DataFrame column-by-column fragments its
    # internal block manager and triggers pandas' PerformanceWarning).
    data = {
        f"{t}_{key}": pd.to_numeric(df.iloc[:, col_for[t][key]], errors="coerce")
        for t in trackers
        for key in wanted_keys
    }
    return pd.DataFrame(data)


def parse_motive_raw(motive_df, sfreq, description=None):
    """read_motive_csv table -> RawArray (misc channels). Positions
    '{tracker}_Position_X' -> '{tracker}_pos_x' converted mm -> m (channel
    unit set to m); rotations '{tracker}_Rotation_X' -> '{tracker}_rot_x'
    kept in degrees."""
    names = [re.sub(r"_(Position|Rotation)_([XYZ])$", lambda m: f"_{m[1][:3].lower()}_{m[2].lower()}", c)
             for c in motive_df.columns]
    is_pos = np.array(["_pos_" in n for n in names])
    data = motive_df.to_numpy(dtype=float).T
    data[is_pos] /= 1000.0
    raw = mne.io.RawArray(data, mne.create_info(names, sfreq=float(sfreq), ch_types="misc"), verbose=False)
    for ch, pos in zip(raw.info["chs"], is_pos):
        if pos:
            ch["unit"] = mne.io.constants.FIFF.FIFF_UNIT_M
    raw.info["description"] = description
    return raw


# ---- get ----------------------------------------------------------------------

def get_pos_channels(tracker, axes):
    """Position channel names of tracker along each axis letter, e.g.
    get_pos_channels("Hip", ("z", "x")) -> ["Hip_pos_z", "Hip_pos_x"];
    tracker None gives bare "pos_{a}" names (single-tracker recordings)."""
    prefix = f"{tracker}_" if tracker else ""
    return [f"{prefix}pos_{a}" for a in axes]


def get_axis_index(axes):
    """Column indices of axis letters in [x, y, z] order."""
    return ["xyz".index(a) for a in axes]


def get_tracker_pos(raw_motion, tracker, axes):
    """Tracker positions along axes -> (n_axes, n_times) in meters."""
    return raw_motion.get_data(picks=get_pos_channels(tracker, axes))


def get_step_state(raw_motion, annot_type="lr_step"):
    """Per-sample swing state from {annot_type}_left / _right annotations:
    -1 left, +1 right, 0 reset or unannotated (index into STEP_COLOR)."""
    sfreq, n = raw_motion.info["sfreq"], raw_motion.n_times
    code = {f"{annot_type}_left": -1, f"{annot_type}_right": 1}
    state = np.zeros(n, dtype=int)
    for a in raw_motion.annotations:
        val = code.get(a["description"].split("/")[0])          # swings may carry "/steplen…"
        if val is not None:
            i0 = max(0, int(round((a["onset"] - raw_motion.first_time) * sfreq)))
            state[i0:min(n, int(round((a["onset"] + a["duration"] - raw_motion.first_time) * sfreq)))] = val
    return state


# ---- annot ---------------------------------------------------------------------

def annot_state_runs(raw_motion, state, labels, min_dur_s, desc=None):
    """Runs of constant state lasting >= min_dur_s -> annotations named
    labels[state] (desc(state, start, length) may extend the name); replaces
    existing annotations with those labels on raw_motion (in place)."""
    sfreq = raw_motion.info["sfreq"]
    starts = np.r_[0, np.flatnonzero(np.diff(state)) + 1]
    lengths = np.diff(np.r_[starts, len(state)])
    keep = lengths >= max(1, int(round(min_dur_s * sfreq)))
    starts, lengths = starts[keep], lengths[keep]
    new = mne.Annotations(onset=starts / sfreq + raw_motion.first_time, duration=lengths / sfreq,
                          description=[desc(state[i], i, n) if desc else labels[state[i]]
                                       for i, n in zip(starts, lengths)],
                          orig_time=raw_motion.info["meas_date"])
    old = raw_motion.annotations
    raw_motion.set_annotations(old[[d.split("/")[0] not in labels.values() for d in old.description]] + new)
    return raw_motion


def annot_lr_step(raw_motion, heading_tracker, ground_axes, lfoot_tracker="LFoot", rfoot_tracker="RFoot",
                  heading_smooth_s=HEADING_SMOOTH_S, speed_smooth_s=0.1, speed_thresh=0.3,
                  min_event_duration_s=0.2):
    """Add lr_step_left / _right / _reset annotations (in place). Each foot's
    ground-plane velocity is projected on the heading of heading_tracker
    (calc_heading) and smoothed over speed_smooth_s; a foot is in swing when
    that forward speed exceeds speed_thresh (m/s):
        left only -> lr_step_left, right only -> lr_step_right, else lr_step_reset.
    Swings carry their step length as "/steplen{m}": the swinging foot's
    ground-plane displacement from first to last sample of the swing."""
    sfreq = raw_motion.info["sfreq"]
    heading = calc_heading(get_tracker_pos(raw_motion, heading_tracker, ground_axes), sfreq, heading_smooth_s)
    feet = {-1: get_tracker_pos(raw_motion, lfoot_tracker, ground_axes),
            1: get_tracker_pos(raw_motion, rfoot_tracker, ground_axes)}
    moving = {side: calc_moving_average((np.gradient(xy, 1 / sfreq, axis=1) * heading).sum(axis=0),
                                        round(speed_smooth_s * sfreq)) > speed_thresh
              for side, xy in feet.items()}
    state = np.zeros(raw_motion.n_times, dtype=int)
    state[moving[-1] & ~moving[1]] = -1
    state[~moving[-1] & moving[1]] = 1
    labels = {-1: "lr_step_left", 0: "lr_step_reset", 1: "lr_step_right"}

    def desc(val, start, n):
        if val == 0:
            return labels[0]
        d = feet[val][:, start + n - 1] - feet[val][:, start]
        return f"{labels[val]}/steplen{np.hypot(*d):.3f}"

    return annot_state_runs(raw_motion, state, labels, min_event_duration_s, desc)


def annot_gait_lean(raw_motion, tracker, ground_axes, heading_smooth_s=HEADING_SMOOTH_S, lean_smooth_s=0.1,
                    min_event_duration_s=0.3):
    """Add gait_lean_left / _right / _reset annotations (in place): lean is
    the raw path heading minus the smoothed heading (calc_heading with
    heading_smooth_s vs 0), wrapped to [-pi, pi) and smoothed over
    lean_smooth_s; lean > 0 (left of the smoothed path) -> left, < 0 -> right.
    Valid for right-handed ground_axes seen from above (e.g. (z, x), y-up)."""
    sfreq = raw_motion.info["sfreq"]
    xy = get_tracker_pos(raw_motion, tracker, ground_axes)
    angle = lambda u: np.arctan2(u[1], u[0])
    lean = (angle(calc_heading(xy, sfreq, 0)) - angle(calc_heading(xy, sfreq, heading_smooth_s)) + np.pi) % (2 * np.pi) - np.pi
    lean = calc_moving_average(lean, round(lean_smooth_s * sfreq))
    state = np.zeros(raw_motion.n_times, dtype=int)
    state[lean > 0], state[lean < 0] = -1, 1
    return annot_state_runs(raw_motion, state, {-1: "gait_lean_left", 0: "gait_lean_reset", 1: "gait_lean_right"},
                         min_event_duration_s)


# ---- parse ---------------------------------------------------------------------

def parse_gait_cycles(raw_motion, annot_type="lr_step", cycle_min_dur=0.6, cycle_max_dur=1.8):
    """Parse {annot_type}_left/_right swing annotations into gait cycles.

    A cycle runs from a left-swing onset to the next left-swing onset and is
    kept iff a right swing starts in between and cycle_min_dur <= duration
    <= cycle_max_dur (stops stretch the gap and are rejected). Resets are
    ignored. Times stay in raw_motion's annotation frame (s since meas_date);
    shift into another raw with io.calc_iso_shift / io.crop_windows(raw_ref=).

    Returns
    -------
    DataFrame, one row per cycle: onset, duration, right_onset,
    left_step_dur_s, right_step_dur_s, left_step_length_m, right_step_length_m
    (step length from the '/steplen{m}' field of annot_lr_step; NaN otherwise).
    """
    a = raw_motion.annotations
    desc = pd.Series(a.description, dtype=str)
    swings = pd.DataFrame({
        "label": desc.str.split("/").str[0],
        "onset": a.onset, "dur": a.duration,
        "steplen": pd.to_numeric(desc.str.extract(r"/steplen([\d.]+)")[0]),
    }).sort_values("onset")
    L = swings[swings["label"] == f"{annot_type}_left"]
    R = swings[swings["label"] == f"{annot_type}_right"]

    l_on, r_on = L["onset"].to_numpy(), R["onset"].to_numpy()
    j = np.searchsorted(r_on, l_on[:-1], side="right")        # first right swing after each left
    r_next = np.append(r_on, np.inf)[j]
    keep = r_next < l_on[1:]                                   # right swing before the closing left
    cyc = pd.DataFrame({
        "onset": l_on[:-1], "duration": l_on[1:] - l_on[:-1], "right_onset": r_next,
        "left_step_dur_s": L["dur"].to_numpy()[:-1],
        "right_step_dur_s": np.append(R["dur"].to_numpy(), np.nan)[j],
        "left_step_length_m": L["steplen"].to_numpy()[:-1],
        "right_step_length_m": np.append(R["steplen"].to_numpy(), np.nan)[j],
    })[keep]
    cyc = cyc[cyc["duration"].between(cycle_min_dur, cycle_max_dur)].reset_index(drop=True)
    print(f"parse_gait_cycles: {len(cyc)} '{annot_type}' cycle(s) "
          f"({cycle_min_dur}-{cycle_max_dur}s) from {len(l_on)} left swing(s)")
    return cyc


# ---- calc ----------------------------------------------------------------------

def calc_heading(xy, sfreq, smooth_s=HEADING_SMOOTH_S):
    """Unit ground-plane heading per sample, (2, n): gradient of the path xy
    (2, n) moving-averaged over smooth_s (0 -> raw path). Zero where the
    path does not move."""
    v = np.gradient(calc_moving_average(xy, round(smooth_s * sfreq), axis=1), 1 / sfreq, axis=1)
    norm = np.hypot(*v)
    return np.divide(v, norm, out=np.zeros_like(v), where=norm > 1e-12)


def calc_speed(xy, sfreq, smooth_s=SPEED_SMOOTH_S):
    """Path speed per sample, (n,) in xy units / s: |gradient| of xy (2, n),
    moving-averaged over smooth_s (even edge reflection: speed is >= 0)."""
    speed = np.hypot(*np.gradient(xy, 1 / sfreq, axis=1))
    return calc_moving_average(speed, round(smooth_s * sfreq), reflect_type="even")


def calc_rhythmicity(sig, sfreq, centers, win_s=2.0, lag_range_s=(0.3, 2.0)):
    """Gait periodicity of sig (e.g. tracker height) around each center
    sample: peak of the normalized autocorrelation of the demeaned win_s
    window over lags in lag_range_s (~ one step to one stride); 1 =
    perfectly periodic. NaN where the window has < 16 samples."""
    half = max(8, int(round(win_s * sfreq))) // 2
    lag0 = max(1, int(round(lag_range_s[0] * sfreq)))
    r = np.full(len(centers), np.nan)
    for k, c in enumerate(centers):
        seg = sig[max(0, c - half):c + half]
        if len(seg) < 16:
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
        lag1 = min(len(ac), max(lag0 + 1, int(round(lag_range_s[1] * sfreq))))
        r[k] = 0.0 if ac[0] == 0 else (ac[lag0:lag1] / ac[0]).max() if lag1 > lag0 else np.nan
    return r


def calc_trial_trace_matrix(trace, sfreq, t0, windows, duration_s, window_s=0.1):
    """Per-window slices of a per-sample trace (sample 0 at time t0, same
    frame as windows) resampled onto a shared [0, duration_s] axis of
    window_s bins -> (len(windows), round(duration_s / window_s)); absorbs
    small timing jitter between trials."""
    n_bins = round(duration_s / window_s)
    return np.array([interp_vector(trace[round((tmin - t0) * sfreq):round((tmax - t0) * sfreq)], frames=n_bins)
                     for tmin, tmax in windows])


def calc_trial_cycle_matrix(cycle_onsets_s, cycle_values, windows, duration_s,
                       cycle_durs_s=None, window_s=0.1, agg="mean"):
    """
    Bin sparse per-cycle scalar values (e.g. step length, step duration, an
    asymmetry index) onto a common nominal per-trial time axis -- the
    per-cycle-event analog of calc_trial_speed_matrix's continuous-signal
    resampling.

    Each row is one window (tmin, tmax). By default (cycle_durs_s=None) a
    cycle whose onset falls inside [tmin, tmax) is linearly rescaled onto
    [0, duration_s) and dropped into that single nominal bin -- a point
    event. Pass cycle_durs_s to instead paint every bin the cycle's own
    [onset, onset + duration) interval overlaps (rescaled the same way) with
    its value -- e.g. so a step's own swing duration renders as a wide mark
    spanning the time it actually took, rather than a single-bin spike.
    Bins with no contributing cycle are NaN (not 0 -- callers should render
    NaN as a visually distinct "no data" color rather than a real low
    value); bins with more than one contributing cycle are aggregated with
    `agg`.

    Parameters
    ----------
    cycle_onsets_s : array, shape (n_cycles,)
        Cycle onsets on the same raw-relative time base as `windows` (e.g.
        parse_gait_cycles' "onset" / "right_onset").
    cycle_values : array, shape (n_cycles,)
        Per-cycle scalar to bin, same length/order as cycle_onsets_s. Pass
        two side-by-side (onset, value) arrays concatenated together (e.g.
        left + right step length, each at its own side's onset) to pool
        both into one row instead of keeping them in separate matrices.
    windows : list of (tmin, tmax)
        Same time base as raw_motion.annotations onset.
    duration_s : float
        Nominal window duration (s) shared by every row's time axis.
    cycle_durs_s : array, shape (n_cycles,), optional
        Per-cycle interval length (s), same length/order as
        cycle_onsets_s; when given, paints the cycle's whole
        [onset, onset + duration) span instead of a single point.
    window_s : float
        Bin width (s) of the shared time axis; n_bins = round(duration_s / window_s).
    agg : "mean" | "median"
        Aggregator applied when more than one cycle contributes to the same bin.

    Returns
    -------
    mat : np.ndarray, shape (len(windows), n_bins)
    """
    cycle_onsets_s = np.asarray(cycle_onsets_s, dtype=float)
    cycle_values   = np.asarray(cycle_values, dtype=float)
    if cycle_durs_s is not None:
        cycle_durs_s = np.asarray(cycle_durs_s, dtype=float)
    n_bins = round(duration_s / window_s)
    agg_fn = {"mean": np.nanmean, "median": np.nanmedian}[agg]

    mat = np.full((len(windows), n_bins), np.nan)
    for r, (tmin, tmax) in enumerate(windows):
        span = tmax - tmin
        in_win = (cycle_onsets_s >= tmin) & (cycle_onsets_s < tmax)
        if not np.any(in_win):
            continue
        onsets = cycle_onsets_s[in_win]
        vals   = cycle_values[in_win]
        starts = np.clip(((onsets - tmin) / span * n_bins).astype(int), 0, n_bins - 1)
        if cycle_durs_s is None:
            ends = starts
        else:
            ends = np.clip((((onsets + cycle_durs_s[in_win]) - tmin) / span * n_bins).astype(int),
                           0, n_bins - 1)

        bin_hits = {}
        for b0, b1, v in zip(starts, ends, vals):
            for b in range(b0, b1 + 1):
                bin_hits.setdefault(b, []).append(v)
        for b, vs in bin_hits.items():
            mat[r, b] = agg_fn(vs)
    return mat


def calc_bone_directions(rot_deg, axis=(0.0, 1.0, 0.0), order="xyz"):
    """
    Rotate a fixed local axis by each row of Euler angles, returning unit
    world-frame direction vectors -- turns a tracked bone's rot_xyz into a
    drawable orientation.

    Parameters:
        rot_deg (np.ndarray): (n, 3) Euler angles in degrees, Motive's
            exported Rotation X/Y/Z columns (read_motive_csv,
            rotation=True).
        axis (tuple): local axis to rotate, default +Y -- Motive's skeletal
            bone convention has a bone's length running along its own local Y.
        order (str): Euler rotation order matching Motive's export, consumed
            by scipy.spatial.transform.Rotation.from_euler.

    Returns:
        np.ndarray: (n, 3) unit direction vectors; rows with any non-finite
        input angle come back all-NaN.
    """
    from scipy.spatial.transform import Rotation
    rot_deg = np.asarray(rot_deg, dtype=float)
    directions = np.full_like(rot_deg, np.nan)
    valid = np.isfinite(rot_deg).all(axis=1)
    if valid.any():
        r = Rotation.from_euler(order, rot_deg[valid], degrees=True)
        directions[valid] = r.apply(np.asarray(axis, dtype=float))
    return directions


def calc_bone_segments(raw_motion, trackers, tmin=None, tmax=None):
    """
    Raw 3-D pose per tracker per sample over [tmin, tmax], from a full-body
    motion Raw carrying '{tracker}_pos_{x,y,z}' and '{tracker}_rot_{x,y,z}'
    channels (sync_motion2rpi.parse_motive_raw with ROTATION=True). Data
    only -- no bone geometry: drawable segments are built by the plot
    functions, sagittal flattening by project_bones_2d.

    Parameters:
        raw_motion (mne.io.Raw): full-body motion recording.
        trackers (list[str]): tracker names (e.g. configs.FULLBODY_TRACKERS).
        tmin, tmax (float | None): raw-relative seconds (raw_motion.times);
            None keeps that edge.

    Returns:
        dict: {tracker: {"t": (n,) raw-relative seconds, "pos": (n, 3)
        meters, "rot": (n, 3) Euler degrees}}, columns x, y, z (world frame,
        pos_y vertical).
    """
    seg = raw_motion.copy().crop(tmin=tmin, tmax=tmax)
    t = seg.times + seg.first_time

    out = {}
    for tracker in trackers:
        pos_ch = [f"{tracker}_pos_{a}" for a in "xyz"]
        rot_ch = [f"{tracker}_rot_{a}" for a in "xyz"]
        missing = [ch for ch in pos_ch + rot_ch if ch not in seg.ch_names]
        if missing:
            raise ValueError(
                f"tracker '{tracker}' missing channel(s) {missing} -- was this "
                f"fif written with sync_motion2rpi's ROTATION=True for rot_xyz?"
            )
        out[tracker] = {
            "t": t,
            "pos": get_tracker_pos(seg, tracker, "xyz").T,
            "rot": seg.get_data(picks=rot_ch).T,            # degrees
        }
    return out


def project_ground_plane(points_xyz, origin_2d, direction_2d, ground_axes, vertical_axis):
    """
    Flatten world points onto the sagittal plane spanned by a ground-plane
    heading (calc_heading_direction) and the vertical.

    Parameters:
        points_xyz (np.ndarray): (..., 3) world points/vectors, columns
            [x, y, z] (calc_bone_segments' column order).
        origin_2d (np.ndarray): (2,) ground point (ground_axes order) forward
            distance is measured from; zeros to project direction vectors.
        direction_2d (np.ndarray): (2,) or (..., 2) unit heading in
            ground_axes order -- one fixed heading, or one per point.
        ground_axes (tuple[str, str]), vertical_axis (str): axis letters
            (configs.GROUND_AXES / configs.VERTICAL_AXIS).

    Returns:
        forward, height (np.ndarray, np.ndarray): each (...,).
    """
    points_xyz = np.asarray(points_xyz, dtype=float)
    ground = points_xyz[..., get_axis_index(ground_axes)] - np.asarray(origin_2d, dtype=float)
    forward = np.sum(ground * np.asarray(direction_2d, dtype=float), axis=-1)
    height = points_xyz[..., get_axis_index(vertical_axis)[0]]
    return forward, height


def project_bones_2d(bone_segments, direction_2d, origin_2d, ground_axes, vertical_axis,
                     axis=(0.0, 1.0, 0.0), order="xyz"):
    """
    Sagittal (heading x vertical) projection of every tracker's 3-D pose:
    position -> (forward, height), orientation -> one in-plane tilt angle.
    The tilt is the bone axis (rot applied to `axis`, calc_bone_directions)
    projected onto the plane, angle = arctan2(forward, vertical) in degrees:
    0 = bone axis straight up (Motive's neutral pose), + = tilted forward.

    Parameters:
        bone_segments (dict): from calc_bone_segments.
        direction_2d (np.ndarray): (n, 2) per-sample (calc_heading_direction, same
            window) or (2,) fixed heading, ground_axes order.
        origin_2d (np.ndarray): (2,) ground-plane origin for forward distance.
        ground_axes, vertical_axis: forwarded to project_ground_plane.
        axis, order: forwarded to calc_bone_directions (bone-local axis,
            Motive's bone convention -- independent of the world axes).

    Returns:
        dict: {tracker: {"t": (n,), "pos": (n, 2) meters (forward, height),
        "angle": (n,) degrees}}.
    """
    out = {}
    for tracker, bone in bone_segments.items():
        forward, height = project_ground_plane(bone["pos"], origin_2d, direction_2d,
                                               ground_axes, vertical_axis)
        d = calc_bone_directions(bone["rot"], axis=axis, order=order)
        d_forward, d_vertical = project_ground_plane(d, np.zeros(2), direction_2d,
                                                     ground_axes, vertical_axis)
        out[tracker] = {
            "t": bone["t"],
            "pos": np.column_stack([forward, height]),
            "angle": np.degrees(np.arctan2(d_forward, d_vertical)),
        }
    return out


def calc_joint_angle(distal, proximal):
    """
    Sagittal joint angle (degrees) = distal - proximal tilt, wrapped to
    [-180, 180). With Motive's neutral pose at 0 for every bone, e.g. ankle
    = LFoot - LShin reads ~0 standing (configs.JOINT_ANGLES pairs).

    Parameters:
        distal, proximal (dict): tracker entries from project_bones_2d.

    Returns:
        np.ndarray: (n,) degrees.
    """
    diff = distal["angle"] - proximal["angle"]
    return (diff + 180.0) % 360.0 - 180.0


# ---- plot ----------------------------------------------------------------------

def plot_path(xy, c=None, colors=None, bin_n=1, cmap="viridis", clim=None, alpha=0.9, lw=2,
              ax=None, cbar_label=None):
    """Ground-plane path xy (2, n) drawn every bin_n samples (bin means),
    segments colored either by a per-sample value c (binned mean, averaged
    over each segment's two ends; colormap + colorbar) or by per-sample
    colors (binned majority, segment takes its start color; e.g.
    [STEP_COLOR[s] for s in get_step_state(raw)]). Axes in xy units; aspect is
    left to the caller."""
    xy = calc_bin_means(xy, bin_n, axis=1)
    pts = xy.T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    if colors is not None:
        rows = np.asarray(colors, dtype=object)[:xy.shape[1] * bin_n].reshape(-1, bin_n)
        lc = LineCollection(segs, colors=[max(set(r), key=list(r).count) for r in rows[:-1]], linewidths=lw, alpha=alpha)
    else:
        lc = LineCollection(segs, cmap=cmap, linewidths=lw, alpha=alpha)
        if c is not None:
            cb = calc_bin_means(c, bin_n)
            lc.set_array((cb[:-1] + cb[1:]) / 2)
            lc.set_clim(*(clim or np.nanpercentile(cb, [5, 95])))
    ax.add_collection(lc)
    ax.autoscale()
    if c is not None and colors is None:
        plt.colorbar(lc, ax=ax, label=cbar_label)
    return ax


def get_bone_colors(trackers, cmap_left="Reds", cmap_right="Blues", cmap_other="Oranges",
                    shade_range=(0.45, 0.9)):
    """
    One base RGBA per tracker: 'L*' trackers get shades of `cmap_left` (red
    hue), 'R*' get shades of `cmap_right` (blue hue), everything else
    (midline: Head/Neck/Chest/Ab/Hip) gets shades of `cmap_other` (orange
    hue) -- side is readable at a glance by hue, individual trackers within a
    side by shade (`shade_range` avoids each colormap's near-white end).

    Parameters:
        trackers (list[str]): tracker names, e.g. configs.FULLBODY_TRACKERS.
        cmap_left, cmap_right, cmap_other (str): matplotlib colormap names.
        shade_range (tuple[float, float]): colormap sample range per side.

    Returns:
        dict: {tracker: (r, g, b, a)}.
    """
    groups = {"L": [], "R": [], "_": []}
    for t in trackers:
        key = "L" if t.startswith("L") else "R" if t.startswith("R") else "_"
        groups[key].append(t)
    cmaps = {"L": plt.get_cmap(cmap_left), "R": plt.get_cmap(cmap_right), "_": plt.get_cmap(cmap_other)}

    colors = {}
    for key, names in groups.items():
        if not names:
            continue
        shades = (np.linspace(*shade_range, len(names)) if len(names) > 1
                 else np.array([np.mean(shade_range)]))
        for name, s in zip(names, shades):
            colors[name] = cmaps[key](s)
    return colors


def get_bone_legend(colors, trackers):
    """Left / right / midline legend handles (one per side present, in that
    side's median tracker shade of get_bone_colors)."""
    handles = []
    for prefix, label in (("L", "left"), ("R", "right"), ("_", "midline")):
        names = ([t for t in trackers if not t.startswith(("L", "R"))] if prefix == "_"
                else [t for t in trackers if t.startswith(prefix)])
        if not names:
            continue
        mid = names[len(names) // 2]
        handles.append(Line2D([0], [0], color=colors[mid], lw=3, label=label))
    return handles


def get_time_rgba(base, n, alpha_range=(0.0, 1.0)):
    """(n, 4) copies of `base` with alpha ramping over time (first -> last frame)."""
    rgba = np.tile(np.asarray(base, dtype=float), (n, 1))
    rgba[:, 3] = np.linspace(*alpha_range, n) if n > 1 else alpha_range[1]
    return rgba


def plot_skeleton_3d(bone_segments, ground_axes, vertical_axis, ax=None, colors=None, bone_length_m=0.10,
                      axis=(0.0, 1.0, 0.0), order="xyz", alpha_range=(0.15, 1.0),
                      lw=1.0, max_frames=150):
    """
    Interactive 3-D skeleton over time: one fixed-length line per tracker per
    frame, centered on its pos and oriented by rotating `axis` by its rot
    (calc_bone_directions), alpha ramping from
    `alpha_range[0]` (earliest frame) to `alpha_range[1]` (latest). Screen
    x/y/z = ground_axes[0] / ground_axes[1] / vertical_axis, so the room's
    vertical renders vertical and the right-handed ground order (e.g. z, x
    under Motive's y-up) keeps the view un-mirrored.

    Drag-to-rotate needs a real GUI backend (plot_fullbody.py sets QtAgg
    before importing pyplot).

    Parameters:
        bone_segments (dict): from calc_bone_segments
            ({tracker: {"t", "pos" (n,3) m, "rot" (n,3) deg}}).
        ground_axes (tuple[str, str]), vertical_axis (str): axis letters
            (configs.GROUND_AXES / configs.VERTICAL_AXIS).
        ax (Axes3D | None): 3-D axes to draw on; a new figure/axes if None.
        colors (dict | None): {tracker: RGBA}, e.g. get_bone_colors(trackers);
            missing trackers fall back to gray.
        bone_length_m (float): drawn line length.
        axis, order: forwarded to calc_bone_directions.
        alpha_range (tuple[float, float]): alpha at the first vs. last frame drawn.
        lw (float): line width.
        max_frames (int | None): thin to at most this many evenly strided
            frames before drawing; None/0 draws every frame.

    Returns:
        Axes3D
    """
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

    n_full = next(iter(bone_segments.values()))["pos"].shape[0]
    stride = int(np.ceil(n_full / max_frames)) if max_frames and n_full > max_frames else 1   # thin to <= max_frames
    screen = get_axis_index(tuple(ground_axes) + (vertical_axis,))

    all_pts = []
    for tracker, bone in bone_segments.items():
        pos = bone["pos"][::stride]
        half = calc_bone_directions(bone["rot"][::stride], axis=axis, order=order) * (bone_length_m / 2)
        start, end = (pos - half)[:, screen], (pos + half)[:, screen]
        base = colors.get(tracker, (0.5, 0.5, 0.5, 1.0)) if colors else (0.5, 0.5, 0.5, 1.0)
        ax.add_collection3d(Line3DCollection(np.stack([start, end], axis=1),
                                             colors=get_time_rgba(base, len(pos), alpha_range),
                                             linewidths=lw))
        all_pts += [start, end]

    pts = np.concatenate(all_pts, axis=0)
    pts = pts[np.isfinite(pts).all(axis=1)]
    ax.set_xlim(pts[:, 0].min(), pts[:, 0].max())
    ax.set_ylim(pts[:, 1].min(), pts[:, 1].max())
    ax.set_zlim(pts[:, 2].min(), pts[:, 2].max())
    ranges = pts.max(axis=0) - pts.min(axis=0)
    ax.set_box_aspect(tuple(max(r, 1e-6) for r in ranges))
    ax.set_xlabel(f"pos_{ground_axes[0]} (m)")
    ax.set_ylabel(f"pos_{ground_axes[1]} (m)")
    ax.set_zlabel(f"pos_{vertical_axis} — vertical (m)")

    handles = get_bone_legend(colors, list(bone_segments)) if colors else []
    if handles:
        ax.legend(handles=handles, loc="upper left", fontsize=9)
    return ax


def plot_skeleton_2d(bones_2d, ax=None, colors=None, bone_length_m=0.10,
                      alpha_range=(0.15, 1.0), lw=1.0, max_frames=150):
    """
    Sagittal skeleton over time: one fixed-length line per tracker per frame,
    centered on its projected (forward, height) pos and tilted by its
    projected angle (0 = vertical, + = forward). Colors and the time
    alpha-fade match plot_skeleton_3d so both views compare directly.

    Parameters:
        bones_2d (dict): from project_bones_2d
            ({tracker: {"t", "pos" (n,2) m, "angle" (n,) deg}}).
        ax, colors, bone_length_m, alpha_range, lw, max_frames: as in
            plot_skeleton_3d.

    Returns:
        Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    n_full = next(iter(bones_2d.values()))["pos"].shape[0]
    stride = int(np.ceil(n_full / max_frames)) if max_frames and n_full > max_frames else 1   # thin to <= max_frames

    for tracker, bone in bones_2d.items():
        pos = bone["pos"][::stride]
        a = np.radians(bone["angle"][::stride])
        half = np.column_stack([np.sin(a), np.cos(a)]) * (bone_length_m / 2)
        base = colors.get(tracker, (0.5, 0.5, 0.5, 1.0)) if colors else (0.5, 0.5, 0.5, 1.0)
        ax.add_collection(LineCollection(np.stack([pos - half, pos + half], axis=1),
                                         colors=get_time_rgba(base, len(pos), alpha_range),
                                         linewidths=lw))

    ax.autoscale()
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("forward distance along heading (m)")
    ax.set_ylabel("height (m)")

    handles = get_bone_legend(colors, list(bones_2d)) if colors else []
    if handles:
        ax.legend(handles=handles, loc="best", fontsize=9)
    return ax
