
from __future__ import annotations
import logging 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
from typing import Iterable, Optional
import mne

from neuromotion.io import pick_or_reref
from neuromotion.calc import extract_band_power, apply_morlet

def save_fig(path: Path, fig=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    (fig or plt.gcf()).savefig(path, bbox_inches="tight", dpi=300)
    logging.info("Saved: %s", path)
    plt.close(fig or plt.gcf())

def plot_mean_with_sem(x, y_matrix, color='blue', label=None, ax=None):
    """
    Plots a time series with mean and shaded standard error.

    Parameters:
    - x: Array-like, time points.
    - y_matrix: 2D Array-like, 1st dim is timepoints 
    - color: Color for the plot and shading.
    - label: Label for the mean line.
    - ax: Matplotlib Axes object to plot on. If None, uses the current Axes.
    """
    # Calculate mean and standard deviation across columns
    y_mean = np.mean(y_matrix, axis=1)
    y_std = np.std(y_matrix, axis=1) / np.sqrt(y_matrix.shape[1])  # Standard error of the mean

    # Use the provided Axes object or the current Axes
    if ax is None:
        ax = plt.gca()

    # Plot mean and shaded standard deviation with clean edges
    ax.plot(x, y_mean, color=color, label=label, linewidth=2)
    ax.fill_between(x, y_mean - y_std, y_mean + y_std, color=color, alpha=0.2, edgecolor=None)
    if label:
        ax.legend()

def plot_pelvis_path(
    position_data: dict,
    subject: str,
    run_idx: Optional[Iterable[int]] = None,  # None => ALL runs
    ax=None,
):
    run_keys_all = list(position_data.keys())

    # Default behavior: plot all runs
    if run_idx is None:
        run_keys = run_keys_all
    else:
        run_idx = list(run_idx)
        run_keys = [run_keys_all[i] for i in run_idx]

    pelvis_path = np.vstack([position_data[k]["pelvis"] for k in run_keys])

    run_str = run_keys[0] if len(run_keys) == 1 else f"{run_keys[0]}-{run_keys[-1]}"

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    ax.plot(pelvis_path[:, 1], pelvis_path[:, 0], label="Pelvis Path",
            color="b", alpha=0.5, linewidth=1)
    ax.set_xlabel("Z Position", fontsize=12)
    ax.set_ylabel("X Position", fontsize=12)
    ax.set_title(f"{subject}: Pelvis Path ({run_str})", fontsize=12, fontweight="bold")
    ax.axis("equal")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig, ax, run_str

def plot_head(sensor_coord, sensor_val=np.arange(63), ax=None, label_off=True, cmap='viridis'): 
    from matplotlib import tri as tri
    standalone = False
    if ax is None:
        fig, ax = plt.subplots() 
        standalone = True
    triang = tri.Triangulation(sensor_coord[:,0], sensor_coord[:,1])
    contour = ax.tricontourf(triang, sensor_val, levels=100, cmap=cmap)  # Filled contours
    if standalone: 
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label("Value")
    ax.set_xlabel("Left to Right")
    ax.axis('equal')
    ax.set_ylabel("Posterior to Anterior")
    ax.set_title("Sensor number on scalp surface")
    # Remove axis labels
    if label_off:
        ax.set_xticks([])  
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_frame_on(False)  # Optional: removes the axis border
    return ax

def plot_tfr(power_data, times, freqs=None, ax=None, vmin=-5, vmax=5, cmap='jet', y_scale='log2', title=None):
        """
        Plot time-frequency representation data.
        
        Parameters
        ----------
        power_data : array, shape (n_freqs, n_times)
            The power data to plot
        times : array, shape (n_times,)
            Time points in seconds
        freqs : array, shape (n_freqs,)
            Frequency points in Hz
        ax : matplotlib.axes.Axes | None
            The axes to plot on. If None, a new figure and axes will be created
        vmin, vmax : float
            Color scale limits
        cmap : str
            Colormap name
        title : str | None
            Title for the plot
            
        Returns
        -------
        im : matplotlib.image.AxesImage
            The image object
        """        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        if freqs is None: 
            freqs = 2 ** np.arange(0,7,0.1)
            freqs = freqs[freqs <= 90]  # Limit to 90 Hz due to amplifier settings

        if y_scale == 'log2': 
            ymin = np.log2(freqs[0])
            ymax = np.log2(freqs[-1])
        elif y_scale == 'linear':
            ymin = freqs[0]
            ymax = freqs[-1]
        
        # Plot TFR with log2 scale for frequency axis
        if y_scale == 'log2':
            im = ax.imshow(
                power_data,
                origin='lower',
                aspect='auto',
                extent=[times[0], times[-1], ymin, ymax],
                interpolation='bilinear',
                vmin=vmin, vmax=vmax,
                cmap=cmap
            )
        elif y_scale == 'linear':
            im = ax.pcolormesh(
                times, freqs, power_data,
                vmin=vmin, vmax=vmax,
                cmap=cmap,
                shading='gouraud'
            )
        
        if y_scale == 'log2':
            all_tick_freqs = np.array([1, 2, 4, 8, 16, 32, 64, 128])
            ytick_freqs = all_tick_freqs[(all_tick_freqs >= freqs[0]) & (all_tick_freqs <= freqs[-1])]
            yticks = np.log2(ytick_freqs)
            ax.set_yticks(yticks)
            ax.set_yticklabels(ytick_freqs)
        elif y_scale == 'linear':
            # Set y-axis ticks for linear scale
            ytick_freqs = np.arange(freqs[0], freqs[-1], 10)
            ax.set_yticks(ytick_freqs)
            ax.set_yticklabels(ytick_freqs)
        
        # Label axes
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        
        # Add vertical line at time=0
        ax.axvline(x=0, color='w', linestyle='--')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (Z-score)', rotation=270, labelpad=15)

        # Add title if provided
        if title is not None:
            ax.set_title(title)
        return im

def plot_psd_for_raws(raw_path_list, channel_picks=None,
                      fmin=1, fmax=100, n_fft=256, segment_dur=30.0):
    """Plot PSD traces every 30s for each raw in a list.

    Each raw gets a base hue from Set1, with lightness varying across segments.
    Only the first segment of each raw is labeled in the legend.
    """
    def _compute_psd_segment(raw, tmin, tmax):
        cropped = raw.copy().crop(tmin=tmin, tmax=tmax, include_tmax=False)
        spectrum = cropped.compute_psd(method="welch", fmin=fmin, fmax=fmax,
                                        n_fft=n_fft, picks=channel_picks)
        return spectrum.freqs, spectrum.get_data().mean(axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    set1 = plt.cm.Set1

    for run_idx, raw_path in enumerate(raw_path_list):
        base_color = set1(run_idx % set1.N)
        raw = mne.io.read_raw_fif(raw_path, preload=True)
        if channel_picks is None:
            channel_picks = raw.ch_names  # default to all channels 
        else:
            raw = pick_or_reref(raw, channel_picks)  # pick/re-reference as needed
        duration = raw.n_times / raw.info['sfreq']
        n_segments = int(duration // segment_dur)

        for seg_idx in range(n_segments):
            tmin = seg_idx * segment_dur
            tmax = tmin + segment_dur

            # Vary alpha/lightness across segments within this run
            alpha = 1.0 - 0.6 * (seg_idx / max(n_segments - 1, 1))

            freqs, psd = _compute_psd_segment(raw, tmin, tmax)
            psd_db = 10 * np.log10(psd + 1e-12)

            label = f"Run {run_idx}" if seg_idx == 0 else None
            ax.plot(freqs, psd_db, color=base_color, alpha=alpha,
                    linewidth=1.5, label=label)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB)")
    ax.set_title(f"Power Spectrum per {int(segment_dur)}s Segment")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(fmin, fmax)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax

def plot_gait_tfr(
    epochs: list[mne.io.RawArray],
    cycle_info,
    ieeg_picks=None,
    mode="average",
    crop_pad=True, #only for individual mode, always true for average mode since we warp to cycle axis
    n_show=5,
    baseline_mode="zscore",
    baseline=None,
    vmin=-3,
    vmax=3,
    cmap="jet",
    n_interp=250,
    n_jobs=4,
    ax=None
):
    """
    Plot TFR of gait-cycle epochs.

    Parameters
    ----------
    epochs : list of mne.io.RawArray from epoch_gait_cycles
    cycle_info : list of dict from epoch_gait_cycles, read pad_s to crop 
    ieeg_picks : list of str or None
        Channel names to pick and average. None uses all. Able to reref if picks are bipolar pairs.
    mode : 'average' or 'individual'
    n_show : int
        Number of epochs in individual mode.
    baseline_mode : str or None
        Passed to apply_morlet ('zscore', 'mean', 'sd', None).
        Baseline is the pre-pad window.
    n_interp : int
        Time points for normalized cycle axis in average mode.
    """
    if not epochs:
        print("No epochs to plot.")
        return None

    sfreq = cycle_info[0]["sfreq"]
    pad_s = cycle_info[0]["pad_s"]
    pad_samp = int(round(pad_s * sfreq))

    # default freqs matching apply_morlet defaults
    freqs = 2 ** np.arange(2, 7, 0.1)
    freqs = freqs[freqs <= 90]

    if mode == "individual":
        n = min(n_show, len(epochs))
        fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), squeeze=False)

        for i in range(n):
            ep = epochs[i]
            if ieeg_picks is not None:
                ep = pick_or_reref(ep, ieeg_picks)
            data = ep.get_data()  # (n_ch, n_samples)

            tfr = apply_morlet(data, sfreq=sfreq, freqs=freqs, output="power",
                               rescale=baseline_mode, baseline=baseline, n_jobs=n_jobs)
            # (1, n_ch, n_freq, n_samples) -> mean over ch -> (n_freq, n_samples)
            tfr = tfr.squeeze(axis=0).mean(axis=0)

            dur = cycle_info[i]["duration"]
            if crop_pad: 
                times = np.linspace(0, dur, tfr.shape[-1])
                tfr = tfr[:, pad_samp:-pad_samp]
            else:
                times = np.linspace(-pad_s, dur + pad_s, tfr.shape[-1])

            plot_tfr(tfr, times, freqs=freqs, ax=axes[i, 0],
                     vmin=vmin, vmax=vmax, cmap=cmap, 
                     title=f"Cycle {i+1} ({dur:.2f}s)")
            axes[i, 0].axvline(x=0, color="w", linestyle="--", linewidth=1)
            axes[i, 0].axvline(x=dur, color="w", linestyle="--", linewidth=1)

        fig.tight_layout()
        return fig, axes

    elif mode == "average":
        n_freqs = len(freqs)
        tfr_stack = []

        for i, (ep, info) in enumerate(zip(epochs, cycle_info)):
            if ieeg_picks is not None:
                ep = pick_or_reref(ep, ieeg_picks)
            data = ep.get_data()

            tfr = apply_morlet(data, sfreq=sfreq, freqs=freqs, output="power",
                               rescale=baseline_mode, baseline=baseline, n_jobs=n_jobs)
            tfr = tfr.squeeze(axis=0).mean(axis=0)  # (n_freq, n_samples)

            # crop pads (computed with pads to avoid edge effects)
            tfr = tfr[:, pad_samp:-pad_samp]
            n_samp = tfr.shape[-1]

            # interpolate each freq to n_interp points so all cycles align
            tfr_interp = np.zeros((n_freqs, n_interp))
            x_orig = np.linspace(0, 1, n_samp)
            x_norm = np.linspace(0, 1, n_interp)
            for fi in range(n_freqs):
                tfr_interp[fi] = np.interp(x_norm, x_orig, tfr[fi])

            tfr_stack.append(tfr_interp)

        tfr_mean = np.mean(tfr_stack, axis=0)
        t_norm = np.linspace(0, n_interp/sfreq, n_interp)

        fig, ax = plt.subplots(figsize=(10, 5)) if ax is None else (None, ax)
        plot_tfr(tfr_mean, t_norm, freqs=freqs, ax=ax,
                 vmin=vmin, vmax=vmax, cmap=cmap,
                 title=f"Average gait cycle TFR (n={len(tfr_stack)})")
        ax.set_xlabel("Normalized gait cycle")

        return fig, ax

def plot_path_overlay_bandpower(
    raw_motion,
    raw_ieeg,
    freq=(13, 30),
    window_s=0.1,
    motion_xy=("pos_z", "pos_x"),
    ieeg_picks=None,
    ieeg_rescale='zscore',
    method="morlet",
    subplot_title=None,
    ax=None
):
    """
    1) Compute band power on entire iEEG run.
    2) Bin both motion and band power into fixed window_s bins.
    3) Plot pos_x vs pos_z trajectory colored by band power.
    """

    # ------------------------------
    # 1. Compute full-run band power
    # ------------------------------
    if ieeg_picks is None:
        ieeg_picks = mne.pick_types(raw_ieeg.info, seeg=True, ecog=True, eeg=True)
        print('No ieeg_picks provided, defaulting to all channels. Found %d channels.' % len(ieeg_picks))
    else:
        raw_ieeg = pick_or_reref(raw_ieeg, ieeg_picks)

    ieeg_data = raw_ieeg.get_data(picks=ieeg_picks)  # (n_ch, n_time)

    band_power = extract_band_power(
        ieeg_data,
        l_freq=freq[0],
        h_freq=freq[1],
        sfreq=raw_ieeg.info["sfreq"],
        method=method,
        rescale=ieeg_rescale
    )  # (n_ch, n_time)

    # Reduce across channels
    if band_power.ndim == 2:
        band_power = band_power.mean(axis=0)  # (n_time,)

    # ------------------------------
    # 2. Build time bins (no warp)
    # ------------------------------
    start = max(raw_motion.times[0], raw_ieeg.times[0])
    end   = min(raw_motion.times[-1], raw_ieeg.times[-1])

    duration = end - start
    n_win = int(np.floor(duration / window_s))

    # motion picks
    motion_picks = mne.pick_channels(raw_motion.ch_names, include=list(motion_xy))

    xy = []
    pw = []

    for k in range(n_win):
        t0 = start + k * window_s
        t1 = t0 + window_s

        # --- motion window ---
        mot_seg = raw_motion.copy().crop(t0, t1)
        mot_data = mot_seg.get_data(picks=motion_picks)

        x = mot_data[0].mean()/1000.0  # convert mm to m
        y = mot_data[1].mean()/1000.0  # convert mm to m
        xy.append([x, y])

        # --- ieeg window ---
        i0, i1 = raw_ieeg.time_as_index([t0, t1])
        pw.append(band_power[i0:i1].mean())

    xy = np.array(xy)
    pw = np.array(pw)

    # ------------------------------
    # 3. Plot colored trajectory
    # ------------------------------
    pts = xy.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    cvals = (pw[:-1] + pw[1:]) / 2

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    lc = LineCollection(segs, cmap="RdBu_r")
    lc.set_array(cvals)
    lc.set_clim(-3, 3)
    lc.set_linewidth(2)
    alpha = 1 / (1 + np.exp(-np.abs(cvals) + 1)) # sigmoid on absolute zscore value 
    lc.set_alpha(alpha) 

    ax.add_collection(lc)
    ax.autoscale()
    ax.set_xlabel(f"{motion_xy[0]} (m)")
    ax.set_ylabel(f"{motion_xy[1]} (m)")
    if subplot_title is not None:
        ax.set_title(subplot_title)
    plt.colorbar(lc, ax=ax, label="Band Power")
    return ax

def plot_path_overlay_speed(
    raw_motion,
    motion_xy=("pos_z", "pos_x"),   # x=pos_z, y=pos_x
    window_s=0.1,                   # set None to use every sample
    speed_smooth_s=0.2,             # smoothing on speed (seconds)
    cmap="viridis",
    ax=None,
    clim=(0,1.5),                      # m/s
    alpha=0.9,
    subplot_title=None,
):
    """
    Plot walking path colored by walking speed.

    - Uses pos_z as x-axis and pos_x as y-axis (both in mm -> converted to meters).
    - Speed computed from derivative of position: sqrt((dx/dt)^2 + (dy/dt)^2) in m/s.
    - Optional binning into window_s means (like your bandpower plot style).
    """
    sfreq = float(raw_motion.info["sfreq"])

    # picks
    picks = mne.pick_channels(raw_motion.ch_names, include=list(motion_xy))
    if len(picks) != 2:
        raise ValueError(f"Missing channels {motion_xy} in raw_motion.ch_names")

    data = raw_motion.get_data(picks=picks)  # (2, n_time) in mm
    x_mm, y_mm = data[0], data[1]

    # convert to meters
    x = x_mm / 1000.0
    y = y_mm / 1000.0

    # --- speed (m/s) ---
    dt = 1.0 / sfreq
    dx = np.gradient(x, dt)
    dy = np.gradient(y, dt)
    speed = np.sqrt(dx**2 + dy**2)

    # optional smoothing of speed
    if speed_smooth_s and speed_smooth_s > 0:
        win = int(round(speed_smooth_s * sfreq))
        win = max(1, win)
        kernel = np.ones(win) / win
        speed = np.convolve(speed, kernel, mode="same")

    # --- optional binning to match your previous style ---
    if window_s is not None:
        w = int(round(window_s * sfreq))
        w = max(1, w)
        n = (len(x) // w) * w  # trim
        x_b = x[:n].reshape(-1, w).mean(axis=1)
        y_b = y[:n].reshape(-1, w).mean(axis=1)
        s_b = speed[:n].reshape(-1, w).mean(axis=1)
        x, y, speed = x_b, y_b, s_b

    # --- build segments ---
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

    # speed per segment
    cvals = (speed[:-1] + speed[1:]) / 2.0

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    lc = LineCollection(segs, cmap=cmap)
    lc.set_array(cvals)
    lc.set_clim(clim if clim is not None else (cvals.min(), cvals.max()))
    lc.set_linewidth(2)
    lc.set_alpha(alpha) 

    ax.add_collection(lc)
    ax.autoscale()
    ax.set_xlabel(f"{motion_xy[0]} (m)")
    ax.set_ylabel(f"{motion_xy[1]} (m)")
    if subplot_title is not None:
        ax.set_title(subplot_title)
    plt.colorbar(lc, ax=ax, label="speed (m/s)")
    return ax

def plot_path_overlay_rhythmicity(
    raw_motion,
    motion_xy=("pos_z", "pos_x"),   # x=pos_z, y=pos_x (mm -> m)
    rhythm_ch="pos_y",              # compute rhythmicity from this channel
    window_s=0.1,                   # binning for path + color (sec)
    rhythm_win_s=2.0,               # analysis window (sec)
    gait_lag_s=(0.3, 2.0),          # used for autocorr peak search (sec)
    cmap="viridis",
    clim=None,
    alpha=0.9,
    ax=None,
    subplot_title=None,
):
    """
    Plot (pos_z, pos_x) path colored by rhythmicity from pos_y.

    metric="autocorr":
        rhythmicity = max normalized autocorr in lag range gait_lag_s
        (higher = more periodic / stable timing)

    """

    sfreq = float(raw_motion.info["sfreq"])

    # --- get data ---
    picks_xy = mne.pick_channels(raw_motion.ch_names, include=list(motion_xy))
    pick_r = mne.pick_channels(raw_motion.ch_names, include=[rhythm_ch])
    if len(picks_xy) != 2:
        raise ValueError(f"Missing {motion_xy} in motion channels.")
    if len(pick_r) != 1:
        raise ValueError(f"Missing {rhythm_ch} in motion channels.")

    xy_mm = raw_motion.get_data(picks=picks_xy)      # (2, n)
    ysig = raw_motion.get_data(picks=pick_r)[0].astype(float)  # (n,)

    x = xy_mm[0] / 1000.0
    y = xy_mm[1] / 1000.0
    ysig = ysig - np.nanmean(ysig)

    # --- window parameters (in samples) ---
    w = max(1, int(round(window_s * sfreq)))
    rw = max(8, int(round(rhythm_win_s * sfreq)))
    if rw >= len(ysig):
        raise ValueError("rhythm_win_s too long for this recording.")

    # Compute rhythmicity per BIN (aligned with your plotting bins)
    n_bins = (len(ysig) // w)
    n_use = n_bins * w
    x = x[:n_use].reshape(n_bins, w).mean(axis=1)
    y = y[:n_use].reshape(n_bins, w).mean(axis=1)

    r = np.full(n_bins, np.nan, float)

    # Each bin uses a centered rhythm_win_s window on raw samples
    half = rw // 2
    for b in range(n_bins):
        center = b * w + w // 2
        i0 = max(0, center - half)
        i1 = min(len(ysig), center + half)
        seg = ysig[i0:i1]
        if len(seg) < 16:
            continue
        seg = seg - np.mean(seg)


        # normalized autocorr peak in lag range
        ac = np.correlate(seg, seg, mode="full")[len(seg)-1:]
        if ac[0] == 0:
            r[b] = 0.0
            continue
        ac = ac / ac[0]
        lag0 = int(round(gait_lag_s[0] * sfreq))
        lag1 = int(round(gait_lag_s[1] * sfreq))
        lag0 = max(1, lag0)
        lag1 = min(len(ac), max(lag0 + 1, lag1))
        r[b] = np.max(ac[lag0:lag1]) if lag1 > lag0 else np.nan

    # --- build segments & color values ---
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    cvals = (r[:-1] + r[1:]) / 2.0

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    lc = LineCollection(segs, cmap=cmap, linewidth=2, alpha=alpha)
    lc.set_array(cvals)
    if clim is None:
        v0, v1 = np.nanpercentile(cvals, [5, 95])
        lc.set_clim(v0, v1)
    else:
        lc.set_clim(*clim)

    ax.add_collection(lc)
    ax.autoscale()
    ax.set_xlabel(f"{motion_xy[0]} (m)")
    ax.set_ylabel(f"{motion_xy[1]} (m)")
    if subplot_title:
        ax.set_title(subplot_title)

    plt.colorbar(lc, ax=ax, label=f"rhythmicity (autocorr)")
    return ax

def plot_path_overlay_gait_lean(
    raw_motion,
    motion_xy=("pos_z", "pos_x"),
    window_s=0.05,
    cmap_left="#e53935",
    cmap_right="#1e88e5",
    color_reset="#222222",
    ax=None,
    subplot_title=None,
    alpha=0.8,
):
    """
    Plot walking path colored by gait lean annotations already on raw_motion.
    Expects annotations: 'gait_lean_left', 'gait_lean_right', 'gait_lean_reset'.
    """
    sfreq = float(raw_motion.info["sfreq"])
    n_times = raw_motion.n_times

    picks = mne.pick_channels(raw_motion.ch_names, include=list(motion_xy))
    data = raw_motion.get_data(picks=picks)
    x = data[0] / 1000.0
    y = data[1] / 1000.0

    # build per-sample color label from annotations
    color_map = {
        "gait_lean_left": cmap_left,
        "gait_lean_right": cmap_right,
        "gait_lean_reset": color_reset,
    }
    sample_color = np.full(n_times, color_reset, dtype=object)

    for annot in raw_motion.annotations:
        if annot["description"] not in color_map:
            continue
        i_start = int(round(annot["onset"] * sfreq))
        i_end = int(round((annot["onset"] + annot["duration"]) * sfreq))
        i_start = max(0, i_start)
        i_end = min(n_times, i_end)
        sample_color[i_start:i_end] = color_map[annot["description"]]

    # optional binning
    if window_s is not None:
        w = max(1, int(round(window_s * sfreq)))
        n = (len(x) // w) * w
        x = x[:n].reshape(-1, w).mean(axis=1)
        y = y[:n].reshape(-1, w).mean(axis=1)
        # majority color per bin
        sc = sample_color[:n].reshape(-1, w)
        sample_color = np.array([
            max(set(row), key=list(row).count) for row in sc
        ], dtype=object)

    # build segments
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    seg_colors = [sample_color[i] for i in range(len(segs))]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    lc = LineCollection(segs, colors=seg_colors, linewidths=2, alpha=alpha)
    ax.add_collection(lc)
    ax.axis("equal")
    ax.set_xlabel(f"{motion_xy[0]} (m)")
    ax.set_ylabel(f"{motion_xy[1]} (m)")
    if subplot_title is not None:
        ax.set_title(subplot_title)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color=cmap_left, lw=2, label="Left lean"),
        Line2D([0], [0], color=cmap_right, lw=2, label="Right lean"),
        Line2D([0], [0], color=color_reset, lw=2, label="Reset"),
    ], loc="best", fontsize=9)

    return ax


# Example usage
if __name__ == "__main__":
    print('Visualization utilities for Neuromotion project')
