
from __future__ import annotations
import logging 
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Optional

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

def save_fig(path: Path, fig=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    (fig or plt.gcf()).savefig(path, bbox_inches="tight", dpi=300)
    logging.info("Saved: %s", path)
    plt.close(fig or plt.gcf())

# Example usage
if __name__ == "__main__":
    # Example data
    x = np.linspace(0, 10, 100)
    y_matrix = np.array([np.sin(x + phase) for phase in np.linspace(0, np.pi, 5)]).T

    # Plot
    plt.figure(figsize=(8, 5))
    plot_mean_with_sem(x, y_matrix, color='green', label='Mean with SEM')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Time Series with Shaded Standard Deviation')
    plt.show()
