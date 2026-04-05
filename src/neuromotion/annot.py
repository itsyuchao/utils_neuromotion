import mne 
import numpy as np
from matplotlib import pyplot as plt

def annot_gait_cycles(
    raw_motion,
    raw_ieeg,
    cycle_min_dur=0.5,
    cycle_max_dur=1.5,
    pad_s=0.5,
)->tuple[list[mne.io.RawArray], list[dict]]:
    """
    Extract iEEG epochs aligned to valid gait cycles (left-right pairs)
    from gait_lean annotations on raw_motion.

    Returns
    -------
    epochs : list of mne.io.RawArray
        Each element is a short Raw segment (n_channels, n_samples)
        with pad_s pre and post, preserving channel info for pick_channels/get_data.
    cycle_info : list of dict
        Per-epoch metadata.
    """
    sfreq_ieeg = float(raw_ieeg.info["sfreq"])

    # --- collect left and right segments ---
    left_segs = []
    right_segs = []
    for annot in raw_motion.annotations:
        if annot["description"] == "gait_lean_left":
            left_segs.append((annot["onset"], annot["duration"]))
        elif annot["description"] == "gait_lean_right":
            right_segs.append((annot["onset"], annot["duration"]))

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
        t_start = onset - pad_s
        t_end = onset + dur + pad_s

        data_dur = raw_ieeg.n_times / sfreq_ieeg
        if t_start < 0 or t_end > data_dur:
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

    print(f"Extracted {len(epochs)} valid gait cycles "
          f"({cycle_min_dur}-{cycle_max_dur}s) from {len(cycles)} candidates")
    return epochs, cycle_info

def annot_gait_lean(
    raw_motion,
    motion_xy=["pos_z", "pos_x"],
    direction_smooth_s=2.0,
    lean_smooth_s=0.1,
    cadence_range=(0.8, 2.0),
    cadence_check_window_s=0.6,
    min_event_duration_s=0.3,
)->mne.io.RawArray:
    """
    Add gait_lean_left / gait_lean_right / gait_lean_reset annotations
    to raw_motion (in-place) and return it.
    """
    sfreq = float(raw_motion.info["sfreq"])
    dt = 1.0 / sfreq
    n_times = raw_motion.n_times

    data = raw_motion.get_data(picks=motion_xy)
    x = data[0] / 1000.0
    y = data[1] / 1000.0

    # smoothed heading
    smooth_win = max(1, int(round(direction_smooth_s * sfreq)))
    kernel = np.ones(smooth_win) / smooth_win
    x_s = np.convolve(x, kernel, mode="same")
    y_s = np.convolve(y, kernel, mode="same")
    heading_smooth = np.arctan2(np.gradient(y_s, dt), np.gradient(x_s, dt))

    # instantaneous heading
    lean_win = max(1, int(round(lean_smooth_s * sfreq)))
    kernel_l = np.ones(lean_win) / lean_win
    x_l = np.convolve(x, kernel_l, mode="same")
    y_l = np.convolve(y, kernel_l, mode="same")
    heading_inst = np.arctan2(np.gradient(y_l, dt), np.gradient(x_l, dt))

    # signed lean
    lean = np.sin(heading_inst - heading_smooth)

    # periodicity gating
    check_win = int(round(cadence_check_window_s * sfreq))
    half_win = check_win // 2
    is_valid = np.zeros(n_times, dtype=bool)
    zero_crossings = np.where(np.diff(np.sign(lean)) != 0)[0]

    for i in range(half_win, n_times - half_win):
        mask = (zero_crossings >= i - half_win) & (zero_crossings < i + half_win)
        freq = mask.sum() / (2.0 * cadence_check_window_s) # within the cadence check window, how many zero crossings /2 to get cycles
        if cadence_range[0] <= freq <= cadence_range[1]:
            is_valid[i] = True 

    # state: -1 left, 0 reset, +1 right
    state = np.zeros(n_times, dtype=int)
    state[is_valid & (lean > 0)] = -1
    state[is_valid & (lean <= 0)] = 1

    # contiguous runs -> annotations
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
        onset=onsets,
        duration=durations,
        description=descriptions,
        orig_time=raw_motion.info["meas_date"],
    )

    # merge with existing annotations
    if raw_motion.annotations is not None and len(raw_motion.annotations):
        print("Warning: raw_motion already has annotations. Check for conflicts. Not writing new annotations.")
    else:
        raw_motion.set_annotations(gait_annot)

    return raw_motion