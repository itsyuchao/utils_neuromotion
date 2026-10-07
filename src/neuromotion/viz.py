
"""Generic plots shared across streams: mean +/- SEM, TFR images, cycle TFRs."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import mne

from neuromotion.percept import pick_or_reref
from neuromotion.calc import apply_morlet, cycles_to_tfr_stack


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
    ax.plot(x, y_mean, color=color, label=label, linewidth=0.5)
    ax.fill_between(x, y_mean - y_std, y_mean + y_std, color=color, alpha=0.2, edgecolor=None)
    if label:
        ax.legend()

def plot_tfr(power_data, times, freqs=None, ax=None, vmin=-5, vmax=5, cmap='jet',
             y_scale='log2', title=None,
             cluster_alpha=0.05,
             contour_color='k', contour_lw=1.0,
             cluster_kwargs=None, n_jobs=1):
        """
        Plot time-frequency representation data.

        Parameters
        ----------
        power_data : array
            Either 2D (n_freqs, n_times) -- plotted directly --
            or 3D (n_obs, n_freqs, n_times). When 3D, the first axis is treated
            as the observation axis; the mean across observations is plotted,
            and an MNE one-sample cluster permutation test with TFCE is run
            against zero. Cluster boundaries with p < ``cluster_alpha`` are
            drawn as smooth contour outlines on top of the smooth
            bilinear / gouraud background.
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
        cluster_alpha : float
            P-value threshold for considering a cluster significant
            (only used when ``power_data`` is 3D and has >=2 observations).
        contour_color : str
            Color of the significant-cluster outline.
        contour_lw : float
            Line width of the contour outline.
        cluster_kwargs : dict | None
            Extra kwargs forwarded to
            ``mne.stats.permutation_cluster_1samp_test``. Defaults to
            ``dict(threshold=dict(start=0, step=0.2), tail=0, n_jobs=n_jobs,
                   out_type='mask', verbose=False)``.
        n_jobs : int
            Number of jobs for the cluster permutation test.

        Returns
        -------
        im : matplotlib.image.AxesImage
            The image object
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # 3D input -> mean + TFCE cluster correction
        sig_mask = None
        if power_data.ndim == 3:
            obs_data = np.asarray(power_data)
            mean_data = obs_data.mean(axis=0)
            if obs_data.shape[0] >= 2:
                from mne.stats import permutation_cluster_1samp_test
                kw = dict(threshold=dict(start=0, step=0.2), tail=0, n_jobs=n_jobs,
                          out_type='mask', verbose=False)
                if cluster_kwargs:
                    kw.update(cluster_kwargs)
                _, clusters, cluster_pv, _ = permutation_cluster_1samp_test(
                    obs_data, **kw
                )
                sig_mask = np.zeros(mean_data.shape, dtype=bool)
                for cl, p in zip(clusters, np.asarray(cluster_pv).ravel()):
                    if p < cluster_alpha:
                        cl_arr = np.asarray(cl)
                        if cl_arr.dtype == bool and cl_arr.shape == mean_data.shape:
                            sig_mask |= cl_arr
                        else:
                            sig_mask[tuple(cl_arr.T) if cl_arr.ndim == 2 else cl_arr] = True
            power_data = mean_data
        elif power_data.ndim != 2:
            raise ValueError(f"power_data must be 2D or 3D, got {power_data.ndim}D")

        if freqs is None:
            freqs = 2 ** np.arange(0, 7, 0.1)
            freqs = freqs[freqs <= 90]

        y_coords = np.log2(freqs) if y_scale == 'log2' else freqs

        im = ax.pcolormesh(
            times, y_coords, power_data,
            vmin=vmin, vmax=vmax,
            cmap=cmap,
            shading='gouraud',
        )

        if sig_mask is not None and sig_mask.any():
            ax.contour(times, y_coords, sig_mask.astype(float),
                       levels=[0.5], colors=contour_color,
                       linewidths=contour_lw)

        if y_scale == 'log2':
            all_tick_freqs = np.array([1, 2, 4, 8, 16, 32, 64, 128])
            ytick_freqs = all_tick_freqs[(all_tick_freqs >= freqs[0]) & (all_tick_freqs <= freqs[-1])]
            ax.set_yticks(np.log2(ytick_freqs))
            ax.set_yticklabels(ytick_freqs)
        else:
            ytick_freqs = np.concatenate(([freqs[0]], np.arange(20, freqs[-1], 20), [freqs[-1]]))
            ax.set_yticks(ytick_freqs)
            ax.set_yticklabels(ytick_freqs)

        ax.set_ylabel('frequency (Hz)')
        ax.set_xlabel('time (s)')
        ax.axvline(x=0, color='w', linestyle='--')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('power (z)', rotation=270, labelpad=15)

        if title is not None:
            ax.set_title(title)

        return im

def plot_cycles_tfr(
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
    Plot TFR of cycle epochs (gait cycles, cue cycles, or any cycle-based segments).

    Parameters
    ----------
    epochs : list of mne.io.RawArray from io.crop_windows
    cycle_info : list of dict from the same source, read pad_s to crop
    ieeg_picks : list of str or None
        Channel names to pick and average. None uses all. Able to reref if picks are bipolar pairs.
    mode : 'average' or 'individual'
    n_show : int
        Number of epochs in individual mode.
    baseline_mode : str or None
        Passed to apply_morlet ('zscore', 'mean', 'sd', None).
        Baseline is the pre-pad window.
    n_interp : int
        Time points for normalized cycle axis in average mode. The x-axis is
        a normalized cycle on [0, 1] regardless of n_interp or sfreq.
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
        # Explicit pipeline: per-cycle TFR -> 3D stack -> plot_tfr (auto cluster correction).
        tfr_stack, freqs = cycles_to_tfr_stack(
            epochs, cycle_info, ch_name=ieeg_picks,
            freqs=freqs, n_interp=n_interp,
            rescale=baseline_mode, baseline=baseline, n_jobs=n_jobs,
        )
        # Cycles are time-warped to a normalized [0, 1] axis; n_interp only
        # controls resolution, not duration.
        t_norm = np.linspace(0, 1, n_interp)

        fig, ax = plt.subplots(figsize=(10, 5)) if ax is None else (None, ax)
        plot_tfr(tfr_stack, t_norm, freqs=freqs, ax=ax,
                 vmin=vmin, vmax=vmax, cmap=cmap,
                 title=f"Average cycle TFR (n={tfr_stack.shape[0]})")
        ax.set_xlabel("Normalized cycle")

        return fig, ax

# Example usage
if __name__ == "__main__":
    print('Visualization utilities for Neuromotion project')


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

    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD (dB)")
    ax.set_title(f"Power Spectrum per {int(segment_dur)}s Segment")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(fmin, fmax)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax
