from __future__ import annotations
import numpy as np
import pandas as pd
import mne

from pathlib import Path
from neuromotion.io import pick_or_reref, save_fig
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def calc_speed(data, diff_step=1, smoothing=10):
    """
    Compute Euclidean derivatives from x, y coordinates in a 2-column NumPy array.
    Pads the start and end with NaN to maintain the same length.

    Parameters:
        data (np.ndarray): Input 2D NumPy array with two columns (x, y coordinates).
        smoothing (int): Number of discrete values before and after to account for smoothing.

    Returns:
        np.ndarray: 1D array of Euclidean derivatives.
    """
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("Input must be a 2D NumPy array with two columns (x, y coordinates).")

    derivatives = np.empty(data.shape[0], dtype=np.float64)
    derivatives[:] = np.nan  # Initialize with NaN for padding

    # Compute finite differences for Euclidean distance
    dx = data[diff_step:, 0] - data[:-diff_step, 0]
    dy = data[diff_step:, 1] - data[:-diff_step, 1]
    derivatives[diff_step//2:-(diff_step//2 + diff_step%2)] = np.sqrt(dx**2 + dy**2) / 2

    # Smooth with a kernel 
    derivatives = np.convolve(derivatives, np.ones(smoothing)/smoothing, mode='same')
    return derivatives

def calc_speed_from_raw(raw_motion, motion_xy=("pos_z", "pos_x"), speed_smooth_s=0.2):
    """
    Compute walking speed (m/s) from two position channels on an mne Raw.

    Position channels are read in mm and converted to meters; speed is the
    magnitude of the numerical derivative (np.gradient at the raw's sfreq),
    optionally smoothed with a moving-average of length speed_smooth_s (s).

    Returns
    -------
    x, y : np.ndarray, shape (n_times,) -- positions in meters
    speed : np.ndarray, shape (n_times,) -- speed in m/s
    """
    sfreq = float(raw_motion.info["sfreq"])
    picks = mne.pick_channels(raw_motion.ch_names, include=list(motion_xy))
    if len(picks) != 2:
        raise ValueError(f"Missing channels {motion_xy} in raw_motion.ch_names")

    data = raw_motion.get_data(picks=picks)  # (2, n_time) in mm
    x, y = data[0] / 1000.0, data[1] / 1000.0

    dt = 1.0 / sfreq
    dx, dy = np.gradient(x, dt), np.gradient(y, dt)
    speed = np.sqrt(dx**2 + dy**2)

    if speed_smooth_s and speed_smooth_s > 0:
        win = max(1, int(round(speed_smooth_s * sfreq)))
        speed = np.convolve(speed, np.ones(win) / win, mode="same")

    return x, y, speed


def calc_path_directions(data, smoothing=10):
    """
    Compute the direction of the path in radians using displacement in x and y directions.
    Handles cases where movement is predominantly in a straight line.

    Parameters:
        data (np.ndarray): Input 2D NumPy array with two columns (x, y coordinates).
        smoothing (int): Number of points to use for smoothing.

    Returns:
        np.ndarray: 1D array of directions in radians.
    """
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("Input must be a 2D NumPy array with two columns (x, y coordinates).")

    directions = np.empty(data.shape[0], dtype=np.float64)
    directions[:] = np.nan

    # Compute displacement in x and y directions
    dx = data[smoothing:, 0] - data[:-smoothing, 0]
    dy = data[smoothing:, 1] - data[:-smoothing, 1]

    # Calculate directions for the valid range
    valid_directions = np.arctan2(dy, dx)
    valid_directions = (valid_directions + 2 * np.pi) % (2 * np.pi)
    pad = smoothing // 2
    directions[pad:pad + len(valid_directions)] = valid_directions

    return directions

def interp_vector(column_vector, frames=250):
    """
    Resample a column vector to match a target size determined by duration and sampling rate.

    Parameters:
        column_vector (np.ndarray): Input 1D NumPy array (column vector).
        sampling_rate (float): Sampling rate in Hz.

    Returns:
        np.ndarray: Resampled 1D NumPy array with the target size.
    """
    target_size = int(frames)
    original_indices = np.linspace(0, len(column_vector) - 1, num=len(column_vector))
    target_indices = np.linspace(0, len(column_vector) - 1, num=target_size)

    resampled_vector = np.interp(target_indices, original_indices, column_vector)
    return resampled_vector

def trial_speed_matrix(raw_motion, windows, duration_s, motion_xy=("pos_z", "pos_x"),
                       speed_smooth_s=0.2, window_s=0.1):
    """
    Stack per-window speed traces from one raw into an (n_windows, n_bins)
    matrix. Each row is resampled (via interp_vector) onto a common axis
    spanning [0, duration_s], so windows of slightly different recorded
    length (timing jitter) still align to the same nominal time axis.

    Parameters
    ----------
    windows : list of (tmin, tmax)
        Same time base as raw_motion.annotations onset (i.e. relative to
        raw_motion.first_time).
    duration_s : float
        Nominal window duration (s) shared by every row's time axis.
    window_s : float
        Bin width (s) of the shared time axis; n_bins = round(duration_s / window_s).

    Returns
    -------
    mat : np.ndarray, shape (len(windows), n_bins)
    """
    sfreq = float(raw_motion.info["sfreq"])
    _, _, speed = calc_speed_from_raw(raw_motion, motion_xy=motion_xy, speed_smooth_s=speed_smooth_s)
    n_bins = round(duration_s / window_s)

    rows = []
    for tmin, tmax in windows:
        i0 = round((tmin - raw_motion.first_time) * sfreq)
        i1 = round((tmax - raw_motion.first_time) * sfreq)
        rows.append(interp_vector(speed[i0:i1], frames=n_bins))
    return np.array(rows)


def cycle_info_to_df(cycle_info):
    """Tidy per-cycle DataFrame from annot_gait_cycles' cycle_info list (one
    dict per gait cycle) -- one row per cycle with cycle_onset_s / cycle_dur_s
    (the left swing's onset/full-stride duration) and right_onset_s (the
    right swing's own onset, reconstructed from cycle_mid_idx --
    cycle_start_idx converted back to seconds), all raw-relative on the same
    time base as raw_motion.annotations onset. Also carries the per-side step
    metrics annot_gait_cycles reads off the swing annotations:
    left_step_dur_s / right_step_dur_s and left_step_length_m /
    right_step_length_m. Positional 1:1 with cycle_info (and so with the
    epochs list annot_gait_cycles returns alongside it) -- callers needing
    cue tags or other per-cycle metadata concat their own columns alongside
    this frame rather than this function reaching into raw.annotations
    itself, keeping it decoupled from any particular annotation schema."""
    return pd.DataFrame({
        "cycle_onset_s":       [c["onset"]              for c in cycle_info],
        "cycle_dur_s":         [c["duration"]            for c in cycle_info],
        "right_onset_s":       [c["onset"] + (c["cycle_mid_idx"] - c["cycle_start_idx"]) / c["sfreq"]
                                                          for c in cycle_info],
        "left_step_dur_s":     [c["left_step_dur_s"]     for c in cycle_info],
        "right_step_dur_s":    [c["right_step_dur_s"]    for c in cycle_info],
        "left_step_length_m":  [c["left_step_length_m"]  for c in cycle_info],
        "right_step_length_m": [c["right_step_length_m"] for c in cycle_info],
    })


def trial_cycle_matrix(cycle_onsets_s, cycle_values, windows, duration_s,
                       cycle_durs_s=None, window_s=0.1, agg="mean"):
    """
    Bin sparse per-cycle scalar values (e.g. step length, step duration, an
    asymmetry index) onto a common nominal per-trial time axis -- the
    per-cycle-event analog of trial_speed_matrix's continuous-signal
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
        cycle_info_to_df's "cycle_onset_s" / "right_onset_s").
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


def calc_step_length(pelvis, l_foot, r_foot, smoothing=1):
    """
    Compute the projection of foot positions onto the pelvis movement direction.

    Parameters:
        pelvis (np.ndarray): Array of pelvis positions with shape (n, 2) for x, y coordinates.
        l_foot (np.ndarray): Array of left foot positions with shape (n, 3) for x, y, z coordinates.
        r_foot (np.ndarray): Array of right foot positions with shape (n, 3) for x, y, z coordinates.
        smoothing (int): Window size for smoothing the projections.

    Returns:
        tuple: Two 1D arrays containing left and right foot projections onto pelvis direction.
    """
    if pelvis.shape[0] != l_foot.shape[0] or pelvis.shape[0] != r_foot.shape[0]:
        raise ValueError("All input arrays must have the same number of frames.")

    if pelvis.shape[1] != 2:
        raise ValueError("Pelvis data must be 2D (x,y coordinates).")

    if l_foot.shape[1] != 3 or r_foot.shape[1] != 3:
        raise ValueError("Foot data must be 3D (x,y,z coordinates).")
    
    # Step 1: Pelvis velocity direction (frame-by-frame)
    pelvis_dir = np.diff(pelvis, axis=0, prepend=pelvis[0:1]) 
    norms = np.linalg.norm(pelvis_dir, axis=1, keepdims=True) 
    norms[norms == 0] = 1  # Prevent division by zero
    pelvis_dir_norm = pelvis_dir / norms

    # Step 2: Convert pelvis direction and positions to 3D but mask out treadmill area
    pelvis_dir_norm_3d = np.column_stack((pelvis_dir_norm[:, 0], np.zeros(pelvis.shape[0]), pelvis_dir_norm[:, 1]))
    pelvis_3d = np.column_stack((pelvis[:, 0], np.zeros(pelvis.shape[0]), pelvis[:, 1]))
    mask = (np.abs(pelvis[:,0])<0.5)&(np.abs(pelvis[:,1])<0.5)
    pelvis_dir_norm_3d[mask] = [1,0,0] # Set to forward direction if mask is true (around treadmill at origin)

    # Step 3: Egocentric foot position
    l_foot_ego = l_foot - pelvis_3d
    r_foot_ego = r_foot - pelvis_3d

    # Step 4: Projection of foot position onto pelvis direction
    l_step = np.sum(l_foot_ego * pelvis_dir_norm_3d, axis=1)
    r_step = np.sum(r_foot_ego * pelvis_dir_norm_3d, axis=1)

    # Step 5: Smooth the projections
    l_step = np.convolve(l_step, np.ones(smoothing)/smoothing, mode='same')
    r_step = np.convolve(r_step, np.ones(smoothing)/smoothing, mode='same')

    return l_step, r_step

def calc_band_power_traces(
    epochs: mne.Epochs,
    event_key="beep",
    picks=("Fz", "FCz"),
    freq_range=(4, 8),
    t_range=(-1, 2),
    method="morlet",
    rescale="zscore",        # or "zscore" / "sd" / None (must match your baseline_correct)
    baseline=(-1, 0),
    combine_channels="mean",   # "mean" or "separate"
    n_jobs=4,
):
    """
    Works on mne.Epochs to get power traces for select pick channels. 
    Also avoids Edge artifact by adding buffer and cropping.

    Returns:
      times: (n_times,)
      y: if combine_channels="mean" -> (n_times, n_trials)
         if combine_channels="separate" -> dict[ch_name] = (n_times, n_trials)
    """
    # add buffer before and after t_range to avoid edge artifact 
    print("Adding 0.5s buffer pre and post defined time range and cropping to avoid edge artifact.")
    buffer = 0.5 #s
    tmin, tmax = t_range
    l_freq, h_freq = freq_range
    ep = epochs[event_key].copy().pick(list(picks)).crop(tmin=tmin-buffer, tmax=tmax+buffer)

    data = ep.get_data()   # (n_trials, optional: n_channels, n_times)
    times = ep.times
    sfreq = int(ep.info["sfreq"])
    buffer_samples = int(sfreq * buffer) 

    # get raw band power
    bp = extract_band_power(
        data,
        l_freq=l_freq,
        h_freq=h_freq,
        sfreq=sfreq,
        method=method,
        rescale=None,
        baseline=None,
        n_jobs=n_jobs,
    )

    # Crop edges before baseline correct
    times = times[buffer_samples:-buffer_samples]
    bp = bp[..., buffer_samples:-buffer_samples] 

    # Then baseline correct, must be defined in samples 
    baseline_start = (tmin-baseline[0])*sfreq
    baseline_dur = (baseline[1]-baseline[0])*sfreq
    bp = baseline_correct(bp, baseline=(baseline_start, baseline_start+baseline_dur), rescale=rescale)

    if len(picks) == 1: 
        return times, bp.T

    if combine_channels == "mean" and len(picks) != 1:
        y = bp.mean(axis=1).T  # (n_times, n_trials) for your plotter
        return times, y

    if combine_channels == "separate":
        out = {}
        for ci, ch in enumerate(ep.ch_names):
            out[ch] = bp[:, ci, :].T  # (n_times, n_trials)
        return times, out

    raise ValueError("combine_channels must be 'mean' or 'separate'")

def extract_band_power(signal, l_freq, h_freq, sfreq=250, method='morlet', rescale=None, baseline=None, n_jobs=4):
    """
    Extract band power from a n_trial x n_channel x n_sample signal array. n_trial and n_channel can be none.

    Parameters:
        signal (np.ndarray): 3D array containing the signal data (n_trial, n_channel, n_sample).
        l_freq (float): Lower frequency of the band in Hz.
        h_freq (float): Upper frequency of the band in Hz.
        sfreq (float): Sampling frequency of the signal in Hz. Default is 250 Hz.
        method (str): Method to use for power calculation ('morlet' or 'hilbert').
        rescale (str): Output type ('sd', 'zscore', or None). Default is None.

    Returns:
        band_power: Band power in the specified frequency range of same shape as input signal.
    """
    if signal.ndim == 1:
        signal = signal[np.newaxis, np.newaxis, :]
    elif signal.ndim == 2:
        signal = signal[np.newaxis, :, :]
    elif signal.ndim == 3:
        signal = signal
    else:
        raise ValueError("Input signal must be 1-3D array.")
    
    if method == 'morlet': 
        exponents = np.arange(0, 7, 0.1)
        freqs = 2 ** exponents
        freqs = freqs[freqs <= 90]
        freq_indices = np.where((freqs >= l_freq) & (freqs <= h_freq))[0]
        power = apply_morlet(signal, sfreq=sfreq, freqs=freqs[freq_indices], n_jobs=n_jobs)
        band_power = np.mean(power, axis=-2).squeeze()  # Average across selected frequencies
    
    elif method == 'hilbert': 
        from scipy.signal import firwin, filtfilt, hilbert
        # Design FIR bandpass filter
        width = 1  # Transition width in Hz
        filter_order = int(sfreq / width)
        # Make filter order odd for zero-phase filtering
        filter_order += 1 if filter_order % 2 == 0 else 0

        # Create FIR filter coefficients
        b = firwin(filter_order, [l_freq, h_freq], pass_zero='bandpass', fs=sfreq)

        # Filter each channel using zero-phase filtering
        filtered_signal = filtfilt(b, 1.0, signal, axis=-1) 

        # Apply Hilbert transform to get the analytic signal
        analytic_signal = hilbert(filtered_signal, axis=-1)

        # Calculate band power (squared magnitude)
        band_power = np.abs(analytic_signal)**2
        band_power = band_power.squeeze()

    if rescale is not None:
        band_power = baseline_correct(band_power, baseline=baseline, rescale=rescale)
    return band_power # should be same dimension as input signal 

def extract_band_phase(signal, l_freq, h_freq, sfreq=250, method='morlet', n_jobs=4):
    """
    Extract band phase from a n_trial x n_channel x n_sample signal array. n_trial and n_channel can be none.

    Parameters:
        signal (np.ndarray): 3D array containing the signal data (n_trial, n_channel, n_sample).
        l_freq (float): Lower frequency of the band in Hz.
        h_freq (float): Upper frequency of the band in Hz.
        sfreq (float): Sampling frequency of the signal in Hz. Default is 250 Hz.
        method (str): Method to use for phase calculation ('morlet' or 'hilbert').

    Returns:
        band_phase: Band phase in the specified frequency range of same shape as input signal.
    """
    if signal.ndim == 1:
        signal = signal[np.newaxis, np.newaxis, :]
    elif signal.ndim == 2:
        signal = signal[np.newaxis, :, :]
    elif signal.ndim == 3:
        signal = signal
    else:
        raise ValueError("Input signal must be 1-3D array.")
    
    if method == 'morlet': 
        exponents = np.arange(0, 7, 0.1)
        freqs = 2 ** exponents
        freqs = freqs[freqs <= 90]
        freq_indices = np.where((freqs >= l_freq) & (freqs <= h_freq))[0]
        phase = apply_morlet(signal, sfreq=sfreq, output='phase', freqs=freqs[freq_indices], n_jobs=n_jobs)
        band_phase = np.mean(phase, axis=-2).squeeze()  # Average across selected frequencies
    
    elif method == 'hilbert': 
        from scipy.signal import firwin, filtfilt, hilbert
        # Design FIR bandpass filter
        width = 1  # Transition width in Hz
        filter_order = int(sfreq / width)
        # Make filter order odd for zero-phase filtering
        filter_order += 1 if filter_order % 2 == 0 else 0

        # Create FIR filter coefficients
        b = firwin(filter_order, [l_freq, h_freq], pass_zero='bandpass', fs=sfreq)

        # Filter each channel using zero-phase filtering
        filtered_signal = filtfilt(b, 1.0, signal, axis=-1) 

        # Apply Hilbert transform to get the analytic signal
        analytic_signal = hilbert(filtered_signal, axis=-1)

        # Calculate band phase
        band_phase = np.angle(analytic_signal)
        band_phase = band_phase.squeeze()

    return band_phase # should be same dimension as input signal

def interp_cycle(core, n_interp, mid=None):
    """
    Linearly interpolate a 1-D cycle core onto a length-``n_interp``
    normalized axis.

    With ``mid`` (a sample index into ``core``, i.e. cycle_info's
    'cycle_mid_idx' re-expressed relative to 'cycle_start_idx' -- the
    right-step onset from annot_gait_cycles), the two half-cycles are
    interpolated INDEPENDENTLY: ``core[:mid+1]`` onto the first
    ``n_interp//2`` output samples and ``core[mid:]`` onto the remaining
    ``n_interp - n_interp//2``, so the mid sample lands exactly on output
    sample ``n_interp//2`` in every cycle (left-step onset -> 0, right-step
    onset -> 1/2, next left-step onset -> 1). With ``mid=None`` the whole
    core is interpolated uniformly (e.g. annot_cue_cycles info, which has
    no mid anchor).
    """
    if mid is None:
        return np.interp(np.linspace(0, 1, n_interp),
                         np.linspace(0, 1, len(core)), core)
    mid = int(np.clip(mid, 1, len(core) - 2))   # keep both halves non-empty
    half = n_interp // 2
    # endpoint=False keeps the mid sample out of the first half, so it appears
    # exactly once -- as the first sample of the second half, at index half.
    first = np.interp(np.linspace(0, 1, half, endpoint=False),
                      np.linspace(0, 1, mid + 1), core[:mid + 1])
    second = np.interp(np.linspace(0, 1, n_interp - half),
                       np.linspace(0, 1, len(core) - mid), core[mid:])
    return np.concatenate([first, second])


def cycles_to_bandpower_matrix(epochs, cycle_info, ch_name,
                               l_freq, h_freq,
                               n_interp=100,
                               rescale="zscore",
                               method="morlet",
                               n_jobs=4,
                               pbar=None):
    """
    Build a (n_interp, n_cycles) band-power matrix from cycle epoch segments.

    ``epochs`` are NOT time-adjusted: they are full padded segments as
    produced by ``annot_gait_cycles`` / ``annot_cue_cycles``. The pad
    indices live in ``cycle_info`` and are applied AFTER the frequency
    transform to avoid Morlet/Hilbert edge artifacts.

    Per-cycle pipeline (explicit, each step independently meaningful):
        1) pick_or_reref(ep, ch_name)
        2) extract_band_power on the FULL padded segment
        3) crop pads with cycle_info[k]['cycle_start_idx':'cycle_end_idx']
        4) interp_cycle the core onto a length-``n_interp`` axis; when
           cycle_info carries 'cycle_mid_idx' (right-step onset sample,
           annot_gait_cycles) the two halves are interpolated independently
           so that sample is anchored at n_interp//2 in every cycle

    Parameters
    ----------
    epochs : list of mne.io.RawArray
        Padded segments (e.g. from annot_gait_cycles / annot_cue_cycles).
    cycle_info : list of dict
        One per epoch. Must carry 'sfreq', 'cycle_start_idx', 'cycle_end_idx'.
    ch_name : str | list of str
        Channel(s) to pick / re-reference. Multi-channel results are mean-collapsed.
    l_freq, h_freq : float
        Band edges in Hz.
    n_interp : int
        Length of the normalized cycle axis.
    rescale : str | None
        Passed to extract_band_power (e.g. 'zscore').
    method : str
        'morlet' or 'hilbert' -- passed through to extract_band_power.
    pbar : object | None
        Progress hook: any object with an ``update(n)`` method (e.g. a tqdm
        bar, possibly shared across concurrent calls), advanced by 1 after
        each cycle is computed.

    Returns
    -------
    mat : np.ndarray, shape (n_interp, n_cycles)
        Empty (n_interp, 0) if no cycles supplied.
    """
    traces = []
    for ep, ci in zip(epochs, cycle_info):
        ep_picked = pick_or_reref(ep, ch_name)
        data = ep_picked.get_data()  # (n_ch, n_samples_padded)
        power = extract_band_power(
            data, l_freq=l_freq, h_freq=h_freq,
            sfreq=ci["sfreq"], rescale=rescale, method=method, n_jobs=n_jobs,
        )
        if power.ndim == 1:
            power = power[np.newaxis, :]
        # mean across channels if multiple, then trim pads with explicit indices
        core = power.mean(axis=0)[ci["cycle_start_idx"]:ci["cycle_end_idx"]]
        mid = ci.get("cycle_mid_idx")
        traces.append(interp_cycle(
            core, n_interp,
            mid=None if mid is None else mid - ci["cycle_start_idx"]))
        if pbar is not None:
            pbar.update(1)
    if not traces:
        return np.empty((n_interp, 0))
    return np.array(traces).T  # (n_interp, n_cycles)


def cycles_to_tfr_stack(epochs, cycle_info, ch_name=None,
                        freqs=None, n_interp=250,
                        rescale="zscore", baseline=None,
                        n_jobs=4, pbar=None):
    """
    Build a (n_cycles, n_freqs, n_interp) Morlet-TFR stack from cycle epoch segments.

    Same conventions as ``cycles_to_bandpower_matrix``:
    epochs are full padded segments, pads are trimmed AFTER the Morlet
    transform using ``cycle_info`` indices, then each frequency row is
    interpolated to a common length-``n_interp`` axis.

    Per-cycle pipeline (explicit):
        1) pick_or_reref(ep, ch_name)  (skip if ch_name is None)
        2) apply_morlet on the FULL padded segment
        3) mean across channels -> (n_freqs, n_samples_padded)
        4) crop pads with cycle_info[k]['cycle_start_idx':'cycle_end_idx']
        5) interp_cycle each freq row onto a length-``n_interp`` axis; when
           cycle_info carries 'cycle_mid_idx' (right-step onset sample,
           annot_gait_cycles) the two halves are interpolated independently
           so that sample is anchored at n_interp//2 in every cycle

    Parameters
    ----------
    epochs : list of mne.io.RawArray
    cycle_info : list of dict   (must carry 'sfreq', 'cycle_start_idx', 'cycle_end_idx')
    ch_name : str | list of str | None
        None -> use all channels of each epoch (mean across channels).
    freqs : array-like | None
        Defaults to log2-spaced 4..90 Hz (matches plot_tfr default range).
    n_interp : int
        Length of the normalized cycle axis.
    rescale, baseline : passed to apply_morlet.
    pbar : object | None
        Progress hook: any object with an ``update(n)`` method (e.g. a tqdm
        bar, possibly shared across concurrent calls), advanced by 1 after
        each cycle is computed.

    Returns
    -------
    stack : np.ndarray, shape (n_cycles, n_freqs, n_interp)
        Empty (0, n_freqs, n_interp) if no cycles supplied.
    freqs : np.ndarray
        Frequency points used.
    """
    if freqs is None:
        freqs = 2 ** np.arange(2, 7, 0.1)
        freqs = freqs[freqs <= 90]
    freqs = np.asarray(freqs)
    n_freqs = len(freqs)

    stack = []
    for ep, ci in zip(epochs, cycle_info):
        ep_used = pick_or_reref(ep, ch_name) if ch_name is not None else ep
        data = ep_used.get_data()  # (n_ch, n_samples_padded)
        tfr = apply_morlet(
            data, sfreq=ci["sfreq"], freqs=freqs, output="power",
            rescale=rescale, baseline=baseline, n_jobs=n_jobs,
        )
        # (1, n_ch, n_freqs, n_samples_padded) -> (n_freqs, n_samples_padded)
        tfr = tfr.squeeze(axis=0).mean(axis=0)
        # explicit pad trim using cycle_info indices
        tfr = tfr[:, ci["cycle_start_idx"]:ci["cycle_end_idx"]]
        mid = ci.get("cycle_mid_idx")
        mid_rel = None if mid is None else mid - ci["cycle_start_idx"]
        tfr_interp = np.zeros((n_freqs, n_interp))
        for fi in range(n_freqs):
            tfr_interp[fi] = interp_cycle(tfr[fi], n_interp, mid=mid_rel)
        stack.append(tfr_interp)
        if pbar is not None:
            pbar.update(1)

    if not stack:
        return np.empty((0, n_freqs, n_interp)), freqs
    return np.array(stack), freqs


def apply_morlet(signal: np.array, sfreq=250, freqs=None, output='power', rescale=None, baseline=None, n_jobs=4, verbose=False):
    """
    Apply Morlet wavelet transform to a signal and return the power spectrum.

    Parameters:
        signal (np.ndarray): 1-3D array containing the signal data (n_trials, n_channel, n_sample).
        sfreq (float): Sampling frequency of the signal in Hz. Default is 250 Hz.
        output (str): Output type ('power', 'phase', or 'complex'). Default is 'power'.
        n_jobs (int): Number of jobs to run in parallel. Default is 4.
        verbose (bool): Whether to print progress information. Default is False.

    Returns:
        np.ndarray: Transformed signal with shape (n_trials, n_channel, n_freq, n_sample).
    """
    from mne.time_frequency import tfr_array_morlet

    if signal.ndim == 1:
        signal = signal[np.newaxis, np.newaxis, :]
    elif signal.ndim == 2:
        signal = signal[np.newaxis, :, :]
    elif signal.ndim == 3:
        signal = signal
    else:
        raise ValueError("Input signal must be 1-3D array.")
    
    if freqs is None:
        exponents = np.arange(0,7,0.1)
        freqs = 2 ** exponents
        freqs = freqs[freqs <= 90]  # Limit to 90 Hz due to amplifier settings
        min_cycles = 2  # Minimum number of cycles (for lowest frequencies)
        n_cycles = np.maximum(min_cycles, freqs / 2)
    else:
        if not isinstance(freqs, np.ndarray):
            freqs = np.array(freqs)
        n_cycles = np.maximum(2, freqs / 2)

    power = tfr_array_morlet(
        signal,
        sfreq=sfreq,
        freqs=freqs,
        n_cycles=n_cycles,
        n_jobs=n_jobs,
        output=output,
        verbose=verbose
    ) 

    if rescale is not None: 
        power = baseline_correct(power, baseline=baseline, rescale=rescale)
    return power # n_trials x n_channel x n_freq x n_sample 


def baseline_correct(data, baseline=None, rescale='zscore', axis=-1):
    """
    Apply baseline correction to the data on the last axis by default.

    Parameters:
        data (np.ndarray): Input data array.
        baseline: Either a tuple (start, end) applied to all trials,
                 or an array of shape (n_trials, 2) for trial-specific baselines.
                 Default is None (use entire data range).
        rescale (str): Rescaling method ('sd', 'zscore', 'mean', or None). Default is 'zscore'.
        axis (int): Axis along which to perform correction. Default is -1 (last axis).

    Returns:
        np.ndarray: Baseline-corrected data.
    """
    # Handle the case where baseline is None
    if baseline is None: 
        start, end = 0, data.shape[axis]
        baseline_mean = np.mean(data[..., start:end], axis=axis, keepdims=True)
        baseline_std = np.std(data[..., start:end], axis=axis, keepdims=True)
    # Handle the case where baseline is a tuple [start, end] for all trials
    elif isinstance(baseline, (list, tuple)) and len(baseline) == 2:
        start, end = baseline
        baseline_mean = np.mean(data[..., start:end], axis=axis, keepdims=True)
        baseline_std = np.std(data[..., start:end], axis=axis, keepdims=True)
    
    # Handle the case where baseline is an array of (n_trials, 2)
    elif isinstance(baseline, np.ndarray) and baseline.shape[-1] == 2:
        if baseline.shape[0] != data.shape[0]:
            raise ValueError(f"Number of baseline periods ({baseline.shape[0]}) must match number of trials ({data.shape[0]})")
        
        baseline_mean = np.zeros(data.shape[:-1] + (1,))
        baseline_std = np.zeros(data.shape[:-1] + (1,))
        
        # Apply different baseline period to each trial
        for i in range(baseline.shape[0]):
            start, end = baseline[i]
            baseline_mean[i] = np.mean(data[i, ..., start:end], axis=axis, keepdims=True)
            baseline_std[i] = np.std(data[i, ..., start:end], axis=axis, keepdims=True)
    else:
        raise ValueError("Baseline must be None, a tuple (start, end), or an array of shape (n_trials, 2)")
    
    # Avoid division by zero
    baseline_std[baseline_std == 0] = 1.0
        
    # Apply rescaling
    if rescale == 'sd':
        return data / baseline_std
    elif rescale == 'zscore':
        return (data - baseline_mean) / baseline_std
    elif rescale == 'mean':
        return data - baseline_mean
    return data


def filter_artifact(raw, window=(-0.5, 1.5), discover_advance_s=34, filter_advance_s=0,
                    ch=(1, 2), sd_thresh=5, fig_path=None, fig_suffix=""):
    """
    Remove stereotyped magnet artifacts from a Percept run by template
    subtraction, per channel index in `ch`:
      1. Discover: onsets = samples crossing below -sd_thresh*SD; gather
         `window` (s)-aligned segments, advancing `discover_advance_s` after
         each, and average them into a channel-specific artifact template.
      2. Match: cross-correlate the run with its template, find_peaks above the
         template energy floor, keep peaks >= `filter_advance_s` apart (0 keeps
         every peak, e.g. closely spaced magnet triplets).
      3. Subtract: least-squares (MLE) scale+offset fit of the template at each
         peak; subtract the scaled artifact shape.
    Saves ArtifactDiscover_ch-*/XCorrPeaks/ArtifactPrePost/FilteredTS QC figures
    to '{fig_path}_desc-*_{fig_suffix}.png' when `fig_path` is given.
    Returns the filtered raw (a copy).
    """
    raw = raw.copy()
    sfreq = raw.info["sfreq"]
    names = raw.ch_names
    ch_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    pre_n, post_n = int(-window[0] * sfreq), int(window[1] * sfreq)
    win_n = pre_n + post_n
    seg_t = np.arange(-pre_n, post_n) / sfreq
    t = np.arange(raw.n_times) / sfreq
    adv_disc, adv_filt = int(discover_advance_s * sfreq), int(filter_advance_s * sfreq)

    orig = raw._data.copy()
    res = {}
    for ch_i in ch:
        x0 = orig[ch_i]

        # 1. discover isolated artifacts -> channel template
        thresh = -sd_thresh * x0.std()
        segs, i = [], pre_n
        while i < len(x0) - post_n:
            if x0[i] < thresh:
                segs.append(x0[i - pre_n:i + post_n])
                i += adv_disc  # advance past this artifact before searching again
            else:
                i += 1
        segs = np.array(segs)
        template = segs.mean(axis=0)
        template = template - template.mean()

        # 2. match: xcorr peaks above energy floor, kept filter_advance_s apart
        xc = np.correlate(x0 - x0.mean(), template, mode="same")
        height = 0.5 * (template ** 2).sum()
        peaks, _ = find_peaks(xc, height=height)
        kept, last = [], -adv_filt - 1
        for p in peaks:
            if p - last >= adv_filt:
                kept.append(p)
                last = p

        # 3. MLE scale+offset fit, subtract the scaled artifact shape
        A = np.vstack([template, np.ones_like(template)]).T
        xf = raw._data[ch_i]
        pre_rows, post_rows = [], []
        for p in kept:
            s = p - win_n // 2  # data window aligned with the centered template
            if s < 0 or s + win_n > len(xf):
                continue
            seg = xf[s:s + win_n].copy()
            (a, _b), *_ = np.linalg.lstsq(A, seg, rcond=None)
            post = seg - a * template
            xf[s:s + win_n] = post
            pre_rows.append(seg)
            post_rows.append(post)
        print(f"{names[ch_i]}: {len(segs)} discovered, {len(post_rows)} subtracted.")
        res[ch_i] = dict(segs=segs, xc=xc, height=height, peaks=np.array(peaks),
                         kept=np.array(kept), pre=np.array(pre_rows), post=np.array(post_rows))

    if fig_path is None:
        return raw

    def _save(desc, fig):
        tag = f"_{fig_suffix}" if fig_suffix else ""
        save_fig(Path(f"{fig_path}_desc-{desc}{tag}.png"), fig=fig)

    def _band(ax, rows, color, label):  # individual traces + mean +/- sem
        for r in rows:
            ax.plot(seg_t, r, color=color, alpha=0.2, linewidth=0.2)
        m = rows.mean(axis=0)
        sem = rows.std(axis=0, ddof=1) / np.sqrt(len(rows))
        ax.plot(seg_t, m, color=color, linewidth=1.5, label=f"{label} (n={len(rows)})")
        ax.fill_between(seg_t, m - sem, m + sem, color=color, alpha=0.3)

    # ArtifactDiscover: segments used to build each channel template
    fig, axes = plt.subplots(1, len(ch), figsize=(6 * len(ch), 5), sharex=True)
    for ax, ch_i in zip(np.atleast_1d(axes), ch):
        _band(ax, res[ch_i]["segs"], ch_colors[ch_i], "discovered")
        ax.axvline(0, color="k", linewidth=0.5, linestyle="--")
        ax.set_title(names[ch_i]); ax.set_xlabel("Time from onset (s)"); ax.legend(fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("Amplitude")
    _save("ArtifactDiscover", fig)

    # XCorrPeaks: find_peaks QC per channel xcorr
    fig, axes = plt.subplots(len(ch), 1, figsize=(14, 4 * len(ch)), sharex=True)
    for ax, ch_i in zip(np.atleast_1d(axes), ch):
        r = res[ch_i]
        ax.plot(t, r["xc"], color=ch_colors[ch_i], linewidth=0.5)
        ax.axhline(r["height"], color="k", linewidth=0.8, linestyle="--", label="height thresh")
        ax.plot(t[r["peaks"]], r["xc"][r["peaks"]], "x", color="gray", markersize=5,
                label=f"find_peaks ({len(r['peaks'])})")
        ax.plot(t[r["kept"]], r["xc"][r["kept"]], "o", mfc="none", mec="red", markersize=8,
                label=f"kept ({len(r['kept'])})")
        ax.set_title(names[ch_i]); ax.set_ylabel("Cross-corr"); ax.legend(fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("Time (s)")
    _save("XCorrPeaks", fig)

    # ArtifactPrePost: fitted windows before/after subtraction
    fig, axes = plt.subplots(1, len(ch), figsize=(6 * len(ch), 5), sharex=True)
    for ax, ch_i in zip(np.atleast_1d(axes), ch):
        _band(ax, res[ch_i]["pre"], "gray", "pre")
        _band(ax, res[ch_i]["post"], ch_colors[ch_i], "post")
        ax.axvline(0, color="k", linewidth=0.5, linestyle="--")
        ax.set_title(names[ch_i]); ax.set_xlabel("Time from onset (s)"); ax.legend(fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("Amplitude")
    _save("ArtifactPrePost", fig)

    # FilteredTS: full-run original vs filtered for inspection
    fig, axes = plt.subplots(len(ch), 1, figsize=(14, 4 * len(ch)), sharex=True)
    for ax, ch_i in zip(np.atleast_1d(axes), ch):
        ax.plot(t, orig[ch_i], color="gray", linewidth=0.5, label="original")
        ax.plot(t, raw._data[ch_i], color=ch_colors[ch_i], linewidth=0.5, label="filtered")
        ax.set_title(names[ch_i]); ax.set_ylabel("Amplitude"); ax.legend(fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("Time (s)")
    _save("FilteredTS", fig)

    return raw
