import mne
import numpy as np
from matplotlib import pyplot as plt

from neuromotion.io import assert_iso_synced, assert_iso_overlap

def annot_gait_lean(                                                                          
    raw_motion,                                                                               
    motion_xy=["pos_z", "pos_x"],                                                             
    direction_smooth_s=1.5,                                                                   
    lean_smooth_s=0.1,                                                          
    min_event_duration_s=0.3,                                                                 
) -> mne.io.RawArray:                                                                         
    """                                                                                       
    Add gait_lean_left / gait_lean_right / gait_lean_reset annotations                        
    to raw_motion (in-place) and return it.                                                   
                                                                                                
    Sign convention (2-D cross product T × offset):                                           
        lean > 0  →  position is LEFT  of the smoothed trajectory                               
        lean < 0  →  position is RIGHT of the smoothed trajectory                               
    """                                                                                       
    sfreq = float(raw_motion.info["sfreq"])
    first_time = raw_motion.first_time                                                   
    dt = 1.0 / sfreq                                                                          
    n_times = raw_motion.n_times                                                              
                                                                                                                                                            
    data = raw_motion.get_data(picks=motion_xy)                                                                                                          
    x = data[0]                                                                                                                              
    y = data[1]                                                                                                                            
                                                                                                                                                            
    # ── smoothed heading path (edge-padded to avoid boundary artifacts) ──                                                                              
    smooth_win = max(1, int(round(direction_smooth_s * sfreq)))                                                                                          
    kernel = np.ones(smooth_win) / smooth_win                                                                                                            
    x_s = np.convolve(np.pad(x, smooth_win // 2, mode="reflect", reflect_type='odd'), kernel, mode="valid")[:n_times]                                                           
    y_s = np.convolve(np.pad(y, smooth_win // 2, mode="reflect", reflect_type='odd'), kernel, mode="valid")[:n_times]                                                           
                                                                                                                                                            
    # ── trajectory direction (heading angle of smoothed path) ────────────                                                                            
    dx_s = np.gradient(x_s, dt)                                                                                                                            
    dy_s = np.gradient(y_s, dt)                                                                                                     
    heading = np.arctan2(dy_s, dx_s)  # -pi - pi

    # ── raw heading from unsmoothed data ─────────────────────────────────
    dx_raw = np.gradient(x, dt)
    dy_raw = np.gradient(y, dt)
    heading_raw = np.arctan2(dy_raw, dx_raw)

    # ── angular deviation: how much raw path "leans" off smooth heading ──
    lean = heading_raw - heading
    lean = (lean + np.pi) % (2 * np.pi) - np.pi   # wrap so no extreme spikes due to arctan -pi to pi boundary
    l_smooth_win = max(1, int(round(lean_smooth_s * sfreq)))
    l_kernel = np.ones(l_smooth_win) / l_smooth_win
    lean = np.convolve(np.pad(lean, l_smooth_win // 2, mode="reflect", reflect_type='odd'), l_kernel, mode="valid")[:n_times]                                                                                                                                                                
                                                                                                                                                            
    # ── state: -1 left, +1 right ───────────────────────                                                                               
    state = np.zeros(n_times, dtype=int)                                                                                                                 
    state[(lean > 0)] = -1   # left of path                                                                                                   
    state[(lean < 0)] =  1   # right of path    

    # ── contiguous runs → annotations ───────────────────────────────────                                                                               
    labels = {-1: "gait_lean_left", 0: "gait_lean_reset", 1: "gait_lean_right"}                                                                          
    min_samp = max(1, int(round(min_event_duration_s * sfreq)))                                                                                          
                                                                                                                                                            
    onsets, durations, descriptions = [], [], []                                                                                                         
    run_start = 0                                                                                                                                        
    run_val = state[0]                                      

    for i in range(1, n_times):                                                               
        if state[i] != run_val or i == n_times - 1:                                           
            run_len = i - run_start                                                           
            if i == n_times - 1 and state[i] == run_val:                                      
                run_len += 1                                                                  
            if run_len >= min_samp:                                                           
                onsets.append(run_start / sfreq)                                              
                durations.append(run_len / sfreq)                                             
                descriptions.append(labels[run_val])                                          
            run_start = i                                                                     
            run_val = state[i]                                                             
                                                                                                
    gait_annot = mne.Annotations(                                                             
        onset=[o + first_time for o in onsets],                                                                         
        duration=durations,                                                                   
        description=descriptions,                                                             
        orig_time=raw_motion.info["meas_date"],                                               
    )                                                                                         
                                                                                                                                                            
    # ── merge: replace any existing gait_lean_* annotations ─────────────                                                                               
    if raw_motion.annotations is not None and len(raw_motion.annotations):                                                                               
        replace_desc = set(gait_annot.description)                                                                                                       
        keep_mask = [d not in replace_desc for d in raw_motion.annotations.description]                                                                  
        kept_annots = raw_motion.annotations[keep_mask]                                                                                                  
        raw_motion.set_annotations(kept_annots + gait_annot)                                                                                             
    else:                                                                                                                                                
        raw_motion.set_annotations(gait_annot)                                                                                                           
                                                                                                                                                            
    return raw_motion


def annot_lr_step(
    raw_motion,
    head_xy=["Handshake_pos_z", "Handshake_pos_x"],
    lfoot_xy=["LFoot_pos_z", "LFoot_pos_x"],
    rfoot_xy=["RFoot_pos_z", "RFoot_pos_x"],
    direction_smooth_s=1.5,
    speed_smooth_s=0.1,
    speed_thresh=300.0,
    min_event_duration_s=0.2,
) -> mne.io.RawArray:
    """
    Add lr_step_left / lr_step_right / lr_step_reset annotations to
    raw_motion (in-place) and return it. Per-foot "moving forward vs not"
    detection using each foot's velocity projected onto the smoothed
    head heading direction (same heading as annot_gait_lean):

        fwd_l > thresh & fwd_r ≤ thresh → lr_step_left   (LFoot in swing)
        fwd_r > thresh & fwd_l ≤ thresh → lr_step_right  (RFoot in swing)
        otherwise (both still or both moving)            → lr_step_reset

    Parameters
    ----------
    speed_thresh : float
        Forward-speed threshold separating moving vs stationary, in data
        units per second. Default 300 suits Motive raw exports (mm/s);
        use ~0.3 for meters.
    """
    sfreq = float(raw_motion.info["sfreq"])
    first_time = raw_motion.first_time
    dt = 1.0 / sfreq
    n_times = raw_motion.n_times

    head  = raw_motion.get_data(picks=head_xy)    # (2, n_times)
    lfoot = raw_motion.get_data(picks=lfoot_xy)
    rfoot = raw_motion.get_data(picks=rfoot_xy)

    # ── smoothed head heading direction (unit vector) ──
    smooth_win = max(1, int(round(direction_smooth_s * sfreq)))
    kernel = np.ones(smooth_win) / smooth_win
    head_s = np.empty_like(head)
    for i in range(2):
        head_s[i] = np.convolve(
            np.pad(head[i], smooth_win // 2, mode="reflect", reflect_type="odd"),
            kernel, mode="valid"
        )[:n_times]
    dhead = np.gradient(head_s, dt, axis=1)
    norm = np.sqrt(dhead[0] ** 2 + dhead[1] ** 2)
    heading_unit = np.zeros_like(dhead)
    nonzero = norm > 1e-9
    heading_unit[:, nonzero] = dhead[:, nonzero] / norm[nonzero]

    # ── per-foot forward speed: velocity projected on heading ──
    dl = np.gradient(lfoot, dt, axis=1)
    dr = np.gradient(rfoot, dt, axis=1)
    fwd_l = dl[0] * heading_unit[0] + dl[1] * heading_unit[1]
    fwd_r = dr[0] * heading_unit[0] + dr[1] * heading_unit[1]

    f_smooth_win = max(1, int(round(speed_smooth_s * sfreq)))
    f_kernel = np.ones(f_smooth_win) / f_smooth_win
    fwd_l = np.convolve(
        np.pad(fwd_l, f_smooth_win // 2, mode="reflect", reflect_type="odd"),
        f_kernel, mode="valid"
    )[:n_times]
    fwd_r = np.convolve(
        np.pad(fwd_r, f_smooth_win // 2, mode="reflect", reflect_type="odd"),
        f_kernel, mode="valid"
    )[:n_times]

    # ── state from per-foot forward-moving vs not ──
    moving_l = fwd_l > speed_thresh
    moving_r = fwd_r > speed_thresh

    state = np.zeros(n_times, dtype=int)
    state[moving_l & ~moving_r] = -1   # LFoot swing → lr_step_left
    state[~moving_l & moving_r] =  1   # RFoot swing → lr_step_right
    # both moving / both still → 0 → lr_step_reset

    # ── contiguous runs → annotations ──
    # Left/right swing annotations additionally carry that swing's step LENGTH
    # as a "/steplen{m}" description field (same "/"-separated scheme as the cue
    # annotations): the net Euclidean (x,z) displacement of the swinging foot
    # (LFoot for left, RFoot for right) between the window's first and last
    # sample. Motive raw exports are in mm, so divide by 1000 to log meters,
    # rounded to 3 dp. Reset (double support) has no swinging foot and stays a
    # bare "lr_step_reset".
    labels = {-1: "lr_step_left", 0: "lr_step_reset", 1: "lr_step_right"}
    min_samp = max(1, int(round(min_event_duration_s * sfreq)))

    def _make_desc(val, start, length):
        if val == -1:
            foot = lfoot
        elif val == 1:
            foot = rfoot
        else:
            return labels[val]   # reset: no swinging foot, no step length
        p0 = foot[:, start]
        p1 = foot[:, start + length - 1]
        step_len_m = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1])) / 1000.0
        return f"{labels[val]}/steplen{step_len_m:.3f}"

    onsets, durations, descriptions = [], [], []
    run_start = 0
    run_val = state[0]

    for i in range(1, n_times):
        if state[i] != run_val or i == n_times - 1:
            run_len = i - run_start
            if i == n_times - 1 and state[i] == run_val:
                run_len += 1
            if run_len >= min_samp:
                onsets.append(run_start / sfreq)
                durations.append(run_len / sfreq)
                descriptions.append(_make_desc(run_val, run_start, run_len))
            run_start = i
            run_val = state[i]

    step_annot = mne.Annotations(
        onset=[o + first_time for o in onsets],
        duration=durations,
        description=descriptions,
        orig_time=raw_motion.info["meas_date"],
    )

    # ── merge: replace any existing lr_step_* annotations. Match on the base
    # label (before any "/steplen…" field) so prior runs are cleaned regardless
    # of the step length each logged. ──
    base_labels = set(labels.values())
    if raw_motion.annotations is not None and len(raw_motion.annotations):
        keep_mask = [d.split("/")[0] not in base_labels
                     for d in raw_motion.annotations.description]
        kept_annots = raw_motion.annotations[keep_mask]
        raw_motion.set_annotations(kept_annots + step_annot)
    else:
        raw_motion.set_annotations(step_annot)

    return raw_motion


def annot_gait_cycles(
    raw_motion,
    raw_ieeg,
    annot_type="lr_step",
    cycle_min_dur=0.6,
    cycle_max_dur=1.8,
    pad_s=0.5,
)->tuple[list[mne.io.RawArray], list[dict]]:
    """
    Extract iEEG epochs aligned to valid gait cycles from
    {annot_type}_left / {annot_type}_right annotations on raw_motion.

    A cycle spans one full stride: it starts at a left swing, includes the
    following right swing, and ends at the onset of the NEXT left swing. The
    cycle duration is that left-onset -> next-left-onset gap, and a cycle is
    accepted only when cycle_min_dur <= duration <= cycle_max_dur (so a stop,
    which stretches the gap past cycle_max_dur, is rejected without any
    separate gap parameter). Any {annot_type}_reset (double-support)
    annotations are ignored entirely; only the left/right swing onsets define
    the cycle. The same left -> right -> next-left walk applies to every
    annot_type (e.g. "lr_step", "gait_lean").

    raw_motion and raw_ieeg need only OVERLAP in ISO wallclock time; they are
    separate runs that share neither start wallclock nor duration (relaxed
    from the previous exact-ISO-sync requirement). ``meas_date`` is the
    authoritative timestamp: motion-frame annotation onsets are re-expressed in
    the ieeg meas_date frame via the two raws' ``meas_date`` difference, and
    only cycles whose padded window falls inside raw_ieeg are kept — so the
    overlapping span alone yields epochs.

    Parameters
    ----------
    annot_type : str
        Prefix of the left/right annotations to consume. Use "gait_lean"
        for output of annot_gait_lean, "lr_step" for annot_lr_step.
    ensure_annot : bool
        If True and annot_type == "lr_step", generate the lr_step_left/right
        annotations on raw_motion (via annot_lr_step) when they are missing,
        so a motion fif loaded straight from disk can be segmented directly.

    Returns
    -------
    epochs : list of mne.io.RawArray
        Each element is a short Raw segment (n_channels, n_samples)
        with pad_s pre and post, preserving channel info for pick_channels/get_data.
    cycle_info : list of dict
        Per-epoch metadata. In addition to the timing/pad-index fields, each
        cycle carries per-side step metrics read straight off the swing
        annotations: ``left_step_dur_s`` / ``right_step_dur_s`` (the left/right
        swing window lengths) and ``left_step_length_m`` / ``right_step_length_m``
        (the step lengths logged by annot_lr_step in the "/steplen{m}"
        description field; NaN for annot_type without that field, e.g.
        "gait_lean").
    """
    # Relaxed constraint: raw_motion and raw_ieeg only need to OVERLAP in ISO
    # wallclock time (not share start + duration). meas_date is authoritative,
    # so a motion-frame annotation onset (seconds since motion meas_date) is
    # re-expressed in the ieeg meas_date frame by adding the meas_date
    # difference. Downstream crop and cycle_info['onset'] then live in the
    # ieeg meas_date frame and compare directly against raw_ieeg.annotations.
    assert_iso_overlap(raw_motion, raw_ieeg, labels=["raw_motion", "raw_ieeg"])

    sfreq_ieeg = float(raw_ieeg.info["sfreq"])
    ieeg_first = raw_ieeg.first_time
    left_desc = f"{annot_type}_left"
    right_desc = f"{annot_type}_right"

    # Shift from the motion meas_date frame to the ieeg meas_date frame. Both
    # onsets are seconds-since-respective-meas_date; under exact ISO sync this
    # equals the old (-motion_first + ieeg_first) anchor swap.
    meas_delta_s = (raw_motion.info["meas_date"]
                    - raw_ieeg.info["meas_date"]).total_seconds()

    def _step_len_m(desc):
        # Step length logged by annot_lr_step in the "/steplen{m}" field, e.g.
        # "lr_step_left/steplen0.523". NaN when absent (e.g. gait_lean).
        for field in desc.split("/")[1:]:
            if field.startswith("steplen"):
                try:
                    return float(field[len("steplen"):])
                except ValueError:
                    return np.nan
        return np.nan

    # --- collect the left/right swing segments in time order (ieeg meas_date
    # frame), tagged L / R and carrying the logged step length (m). Reset
    # (double-support) annotations are ignored entirely: only swing onsets
    # define a cycle. We walk this merged sequence to grab each cycle as a
    # left swing -> following right swing -> next left swing.
    segs = []   # (onset, duration, kind, step_length_m)
    for annot in raw_motion.annotations:
        onset = annot["onset"] + meas_delta_s  # motion frame -> ieeg frame
        label = annot["description"].split("/")[0]
        if label == left_desc:
            segs.append((onset, annot["duration"], "L", _step_len_m(annot["description"])))
        elif label == right_desc:
            segs.append((onset, annot["duration"], "R", _step_len_m(annot["description"])))
    segs.sort(key=lambda s: s[0])
    n_seg = len(segs)

    # --- walk left -> right -> next-left into cycles. The cycle duration is
    # the onset gap between a left swing and the subsequent left swing (one
    # full stride); the intervening right swing supplies the right-side step
    # metrics. Resets are absent from segs, so only swing onsets are seen. ---
    cycles = []
    i = 0
    while i < n_seg:
        if segs[i][2] != "L":
            i += 1
            continue
        l_on, l_dur, _, l_len = segs[i]

        # scan forward to the next right swing; restart from a later left if
        # one appears before any right.
        j = i + 1
        while j < n_seg and segs[j][2] != "R":
            if segs[j][2] == "L":
                break
            j += 1
        if j >= n_seg or segs[j][2] != "R":
            i = j if (j < n_seg and segs[j][2] == "L") else i + 1
            continue

        _, r_dur, _, r_len = segs[j]

        # scan forward to the closing left swing (start of the next cycle);
        # its onset minus this left's onset is the cycle duration.
        k = j + 1
        while k < n_seg and segs[k][2] != "L":
            k += 1
        if k >= n_seg:
            break   # no closing left swing; the final cycle cannot be formed

        cycle_dur = segs[k][0] - l_on      # left onset -> next left onset
        if cycle_min_dur <= cycle_dur <= cycle_max_dur:
            cycles.append({
                "onset": l_on,
                "duration": cycle_dur,
                "left_step_dur_s": l_dur,
                "right_step_dur_s": r_dur,
                "left_step_length_m": l_len,
                "right_step_length_m": r_len,
            })
        # the closing left swing begins the next cycle
        i = k

    # --- epoch from raw_ieeg with padding ---
    epochs = []
    cycle_info = []

    for cyc in cycles:
        onset, dur = cyc["onset"], cyc["duration"]
        t_start = onset - ieeg_first - pad_s
        t_end = onset - ieeg_first + dur + pad_s

        if t_start < 0 or t_end > raw_ieeg.times[-1]:
            continue

        epoch_raw = raw_ieeg.copy().crop(tmin=t_start, tmax=t_end, include_tmax=False)
        epochs.append(epoch_raw)

        pad_samp = int(round(pad_s * sfreq_ieeg))
        cycle_samp = epoch_raw.n_times - 2 * pad_samp
        cycle_info.append({
            "onset": onset,
            "duration": dur,
            "pad_s": pad_s,
            "sfreq": sfreq_ieeg,
            "cycle_start_idx": pad_samp,
            "cycle_end_idx": pad_samp + cycle_samp,
            "n_samples": epoch_raw.n_times,
            "left_step_dur_s": cyc["left_step_dur_s"],
            "right_step_dur_s": cyc["right_step_dur_s"],
            "left_step_length_m": cyc["left_step_length_m"],
            "right_step_length_m": cyc["right_step_length_m"],
        })

    print(f"Extracted {len(epochs)} valid gait cycles from '{annot_type}' "
          f"({cycle_min_dur}-{cycle_max_dur}s) from {len(cycles)} candidates")
    return epochs, cycle_info


def annot_cue_cycles(
    raw,
    periods,
    cycle_len_s,
    pad_s=0.5,
) -> tuple[list[mne.io.RawArray], list[dict]]:
    """
    Subdivide each absolute-time period into fixed-length cue cycles and
    crop raw segments around them. Output format mirrors annot_gait_cycles:
    a list of padded epoch segments and a list of cycle_info dicts.

    IMPORTANT: the returned epochs are NOT time-adjusted to the cycle core.
    Each epoch spans ``[cycle_start - pad_s, cycle_end + pad_s]`` so that a
    downstream Morlet / Hilbert transform sees the pads as buffer against
    edge artifacts. The cycle_info dict carries ``cycle_start_idx`` and
    ``cycle_end_idx`` which the caller is expected to use AFTER the
    frequency-domain transform to trim the pads. See
    ``cycles_to_bandpower_matrix`` and ``cycles_to_tfr_stack`` for the
    matching downstream steps.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous recording to crop from.
    periods : iterable of (t0, t1)
        Absolute raw times (i.e. including ``raw.first_time``) demarcating
        the parent periods (e.g. Beep_ON / Beep_OFF / baseline blocks).
    cycle_len_s : float
        Core length of each cue cycle (without pads).
    pad_s : float
        Padding before and after each cycle. Should match what the caller
        plans to trim in ``cycle_info['cycle_start_idx':'cycle_end_idx']``.

    Returns
    -------
    epochs : list of mne.io.RawArray
        Padded segments (NOT time-adjusted -- pads are still present).
    cycle_info : list of dict
        Per-cycle metadata: 'sfreq', 'pad_s', 'duration', 'onset',
        'cycle_start_idx', 'cycle_end_idx', 'n_samples', 'period_idx'.
    """
    sfreq = float(raw.info["sfreq"])
    raw_ft = raw.first_time
    pad_samp = int(round(pad_s * sfreq))
    epochs, cycle_info = [], []
    periods = list(periods)

    for i, (t0, t1) in enumerate(periods):
        n_cycles = int(np.floor((t1 - t0) / cycle_len_s))
        for k in range(n_cycles):
            c0 = t0 + k * cycle_len_s
            c1 = c0 + cycle_len_s
            tmin = c0 - raw_ft - pad_s
            tmax = c1 - raw_ft + pad_s
            if tmin < 0 or tmax > raw.times[-1]:
                continue
            ep = raw.copy().crop(tmin=tmin, tmax=tmax, include_tmax=False)
            n_samples = ep.n_times
            cycle_info.append({
                "sfreq": sfreq,
                "pad_s": pad_s,
                "duration": cycle_len_s,
                "onset": c0,
                "cycle_start_idx": pad_samp,
                "cycle_end_idx": n_samples - pad_samp,
                "n_samples": n_samples,
                "period_idx": i,
            })
            epochs.append(ep)

    print(f"Extracted {len(epochs)} cue cycles ({cycle_len_s:.3f}s each) "
          f"from {len(periods)} periods")
    return epochs, cycle_info