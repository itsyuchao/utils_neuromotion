from __future__ import annotations
import numpy as np
import mne

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

def get_band_power_traces(
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

def apply_morlet(signal: np.array, sfreq=250, freqs=None, output='power', rescale=None, baseline=None, n_jobs=4):
    """
    Apply Morlet wavelet transform to a signal and return the power spectrum.

    Parameters:
        signal (np.ndarray): 1-3D array containing the signal data (n_trials, n_channel, n_sample).
        sfreq (float): Sampling frequency of the signal in Hz. Default is 250 Hz.
        output (str): Output type ('power', 'phase', or 'complex'). Default is 'power'.
        n_jobs (int): Number of jobs to run in parallel. Default is 4.

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
        output=output
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

