import mne
import numpy as np
from matplotlib import pyplot as plt

from neuromotion.io import assert_iso_synced

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
    labels = {-1: "lr_step_left", 0: "lr_step_reset", 1: "lr_step_right"}
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

    step_annot = mne.Annotations(
        onset=[o + first_time for o in onsets],
        duration=durations,
        description=descriptions,
        orig_time=raw_motion.info["meas_date"],
    )

    # ── merge: replace any existing lr_step_* annotations ──
    if raw_motion.annotations is not None and len(raw_motion.annotations):
        replace_desc = set(step_annot.description)
        keep_mask = [d not in replace_desc for d in raw_motion.annotations.description]
        kept_annots = raw_motion.annotations[keep_mask]
        raw_motion.set_annotations(kept_annots + step_annot)
    else:
        raw_motion.set_annotations(step_annot)

    return raw_motion


def annot_gait_cycles(
    raw_motion,
    raw_ieeg,
    annot_type="gait_lean",
    cycle_min_dur=0.6,
    cycle_max_dur=1.8,
    pad_s=0.5,
)->tuple[list[mne.io.RawArray], list[dict]]:
    """
    Extract iEEG epochs aligned to valid gait cycles (left-right pairs)
    from {annot_type}_left / {annot_type}_right annotations on raw_motion.

    Parameters
    ----------
    annot_type : str
        Prefix of the left/right annotations to consume. Use "gait_lean"
        for output of annot_gait_lean, "lr_step" for annot_lr_step.

    Returns
    -------
    epochs : list of mne.io.RawArray
        Each element is a short Raw segment (n_channels, n_samples)
        with pad_s pre and post, preserving channel info for pick_channels/get_data.
    cycle_info : list of dict
        Per-epoch metadata.
    """
    # Hard-require ISO wallclock alignment: start wallclock and duration
    # must agree to tolerance. Under that invariant, the motion-frame
    # annotation onset can be re-expressed in ieeg's meas_date frame by
    # swapping the two raws' first_time anchors (this correctly handles the
    # case where meas_date differs but absolute start wallclock matches).
    # Downstream crop and cycle_info['onset'] then live in the ieeg
    # meas_date frame and compare directly against raw_ieeg.annotations.
    assert_iso_synced(raw_motion, raw_ieeg, labels=["raw_motion", "raw_ieeg"])

    sfreq_ieeg = float(raw_ieeg.info["sfreq"])
    motion_first = raw_motion.first_time
    ieeg_first = raw_ieeg.first_time
    left_desc = f"{annot_type}_left"
    right_desc = f"{annot_type}_right"

    # --- collect left and right segments in ieeg meas_date wallclock ---
    left_segs = []
    right_segs = []
    for annot in raw_motion.annotations:
        onset = annot["onset"] - motion_first + ieeg_first  # motion-frame -> ieeg-frame
        if annot["description"] == left_desc:
            left_segs.append((onset, annot["duration"]))
        elif annot["description"] == right_desc:
            right_segs.append((onset, annot["duration"]))

    left_segs.sort(key=lambda x: x[0])
    right_segs.sort(key=lambda x: x[0])

    # --- pair left-right into cycles ---
    used_right = set()
    cycles = []

    for l_on, l_dur in left_segs:
        l_end = l_on + l_dur
        best_ri = None
        best_gap = np.inf
        for ri, (r_on, r_dur) in enumerate(right_segs):
            if ri in used_right:
                continue
            gap = r_on - l_end
            if -0.05 <= gap < best_gap:
                best_gap = gap
                best_ri = ri
        if best_ri is None or best_gap > 0.1:
            continue
        used_right.add(best_ri)
        r_on, r_dur = right_segs[best_ri]
        cycle_dur = (r_on + r_dur) - l_on
        if cycle_min_dur <= cycle_dur <= cycle_max_dur:
            cycles.append((l_on, cycle_dur))

    # --- epoch from raw_ieeg with padding ---
    epochs = []
    cycle_info = []

    for onset, dur in cycles:
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