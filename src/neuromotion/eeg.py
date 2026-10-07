"""Scalp EEG / EXG (EEGLAB .set): electromagnet trigger onsets, montage and
trigger QC plot."""
import matplotlib.pyplot as plt
import mne
import numpy as np

from neuromotion.io import fmt_mmss_mmm


# ---- parse ------------------------------------------------------------------

def parse_eeg_stims(raw, stim_label="s1"):
    """Onsets (s, annotation frame) of the stim_label trigger annotations."""
    return raw.annotations.onset[raw.annotations.description == stim_label]


# ---- get -----------------------------------------------------------------------

def get_antneuro_montage() -> mne.channels.DigMontage:
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


# ---- plot ----------------------------------------------------------------------

def plot_eeg_stims(raw, stim_label="s1"):
    """Stem plot of the stim_label trigger onsets with inter-trigger intervals."""
    onsets = parse_eeg_stims(raw, stim_label)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.stem(onsets, np.ones_like(onsets), linefmt="grey", markerfmt="o", basefmt=" ")
    for i, (prev, t) in enumerate(zip(onsets, onsets[1:]), start=1):
        ax.text(t, (i % 5) * 0.1 + 0.5, f"{fmt_mmss_mmm(t - prev)}s", rotation=45, va="bottom", ha="center", fontsize=8)
    ax.set(ylim=(0, 1.2), xlabel="Time (s)", title=f"EEG Stimulus Markers ({stim_label}) with Inter-Stimulus Durations")
    fig.tight_layout()
    return fig


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
