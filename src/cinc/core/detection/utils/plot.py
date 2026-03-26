"""
This module provides visualization functions for detection results.

Authors: Arnaud Poletto
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict


def _plot_single_phase_on_axis(
    ax,
    signal_data: np.ndarray,
    results: Dict,
    time: np.ndarray,
    color_signal: str,
    color_peak: str,
    color_filled: str,
    color_grid: str,
    color_text: str,
    phase_name: str = None,
) -> None:
    """
    Plot single phase detection results on a given axis.

    Args:
        ax: Matplotlib axis object.
        signal_data (np.ndarray): The signal data to plot.
        results (Dict): Detection results.
        time (np.ndarray): Time array.
        color_signal (str): Color for signal line.
        color_peak (str): Color for peaks.
        color_filled (str): Color for filled detections.
        color_grid (str): Color for grid.
        color_text (str): Color for text.
        phase_name (str): Optional phase name for title.
    """
    # Plot signal with subtle styling
    ax.plot(
        time,
        signal_data,
        color=color_signal,
        linewidth=2,
        alpha=0.6,
        zorder=2,
        label="Signal",
    )

    # Extract detection data
    detections = results["detections"]
    are_filled = results.get("are_filled", None)
    intervals = results.get("intervals", None)

    # Plot intervals if available
    if intervals is not None and len(intervals) > 0:
        interval_indices = detections[intervals]
        for i, (start_idx, end_idx) in enumerate(interval_indices):
            start_time = time[start_idx]
            end_time = time[end_idx]

            ax.axvspan(
                start_time,
                end_time,
                alpha=0.12,
                color=color_filled,
                zorder=1,
                label="Event Cycle" if i == 0 else "",
            )

    # Plot detections
    if len(detections) == 0:
        # Empty result handling
        ax.text(
            0.5,
            0.5,
            "⚠️  No detections found",
            transform=ax.transAxes,
            fontsize=16,
            ha="center",
            va="center",
            color=color_text,
            fontweight="bold",
            alpha=0.5,
        )
    else:
        # Plot with filled/original distinction
        if are_filled is not None and np.any(are_filled):
            filled_detections = detections[are_filled]
            original_detections = detections[~are_filled]

            # Filled detections
            if len(filled_detections) > 0:
                ax.scatter(
                    time[filled_detections],
                    signal_data[filled_detections],
                    color=color_filled,
                    s=180,
                    zorder=6,
                    marker="X",
                    edgecolors="white",
                    linewidths=2.5,
                    alpha=0.9,
                    label=f"Filled Detections ({len(filled_detections)})",
                )

            # Original detections
            if len(original_detections) > 0:
                ax.scatter(
                    time[original_detections],
                    signal_data[original_detections],
                    color=color_peak,
                    s=120,
                    zorder=5,
                    marker="o",
                    edgecolors="white",
                    linewidths=2.5,
                    label=f"Detected Peaks ({len(original_detections)})",
                )
        else:
            # All detections
            ax.scatter(
                time[detections],
                signal_data[detections],
                color=color_peak,
                s=120,
                zorder=5,
                marker="o",
                edgecolors="white",
                linewidths=2.5,
                label=f"Detected Peaks ({len(detections)})",
            )

        # Calculate and display statistics
        if len(detections) > 1:
            duration_min = time[-1] / 60
            event_rate = len(detections) / duration_min if duration_min > 0 else 0

            stats_text = (
                f"Total Peaks: {len(detections)}\n"
                f"Rate: {event_rate:.1f} peaks/min\n"
                f"Duration: {time[-1]:.1f}s"
            )
            ax.text(
                0.0,
                -0.1,
                stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(
                    boxstyle="round,pad=0.6",
                    facecolor="white",
                    edgecolor=color_text,
                    linewidth=2,
                    alpha=0.95,
                ),
                color=color_text,
                fontfamily="monospace",
                fontweight="bold",
            )

    # Enhanced styling
    ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold", color=color_text)
    ax.set_ylabel("Amplitude", fontsize=12, fontweight="bold", color=color_text)

    # Title
    if phase_name:
        title = f"Detection - {phase_name}"
    else:
        title = "Detection - Peak Analysis"
    ax.set_title(title, fontsize=15, fontweight="bold", color=color_text, pad=15)

    # Enhanced legend with better positioning (bottom right, outside)
    legend = ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, -0.05),
        fontsize=10,
        framealpha=0.95,
        edgecolor=color_text,
        fancybox=True,
        shadow=False,
        borderpad=1,
        labelspacing=0.8,
    )
    legend.get_frame().set_linewidth(2)

    # Sophisticated grid
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=1, color=color_grid, zorder=0)
    ax.set_xlim(0, time[-1])

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(color_text)
    ax.spines["bottom"].set_color(color_text)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    # Tick styling
    ax.tick_params(colors=color_text, which="both", labelsize=10)


def plot_detections(
    signal_data: np.ndarray | List[np.ndarray],
    results: Dict | List[Dict],
    fs: float,
) -> None:
    """
    Plot detection results for physiological signals.

    This function creates a professional visualization of detection results
    (respiratory or cardiac) with clear visual hierarchy, color-coded phases, and statistical information.

    Args:
        signal_data (np.ndarray | List[np.ndarray]): The signal data or list of signals for each phase.
        results (Dict | List[Dict]): Detection results containing detections, prominences, and intervals.
        fs (float): The sampling frequency in Hz.
    """
    # Check if we have both phases or single phase
    is_both_phases = isinstance(results, list)
    is_signal_list = isinstance(signal_data, list)

    # Define color palette
    color_signal = "#7f8c8d"
    color_peak = "#e74c3c"
    color_trough = "#3498db"
    color_filled = "#2ecc71"
    color_grid = "#bdc3c7"
    color_text = "#34495e"

    # Get signal data for time array
    signal_for_time = signal_data[0] if is_signal_list else signal_data
    time = np.arange(len(signal_for_time)) / fs

    # Case 1: Both phases with separate signals
    if is_both_phases and is_signal_list:
        fig, axes = plt.subplots(2, 1, figsize=(20, 12), sharex=True)
        fig.patch.set_facecolor("white")

        # Plot positive phase
        _plot_single_phase_on_axis(
            ax=axes[0],
            signal_data=signal_data[0],
            results=results[0],
            time=time,
            color_signal=color_signal,
            color_peak=color_peak,
            color_filled=color_filled,
            color_grid=color_grid,
            color_text=color_text,
            phase_name="Positive Phase",
        )

        # Plot negative phase
        _plot_single_phase_on_axis(
            ax=axes[1],
            signal_data=signal_data[1],
            results=results[1],
            time=time,
            color_signal=color_signal,
            color_peak=color_trough,
            color_filled=color_filled,
            color_grid=color_grid,
            color_text=color_text,
            phase_name="Negative Phase",
        )

        plt.tight_layout(rect=[0, 0.02, 1, 1])
        plt.show()
        return

    # Case 2: Both phases with single signal
    if is_both_phases and not is_signal_list:
        fig, ax = plt.subplots(figsize=(20, 7))
        fig.patch.set_facecolor("white")

        # Plot signal with subtle styling
        ax.plot(
            time,
            signal_data,
            color=color_signal,
            linewidth=2,
            alpha=0.6,
            zorder=2,
            label="Signal",
        )

        # Extract phase data
        pos_detections = results[0]["detections"]
        neg_detections = results[1]["detections"]
        pos_prominences = results[0]["prominences"]
        neg_prominences = results[1]["prominences"]
        pos_are_filled = results[0].get("are_filled", None)
        neg_are_filled = results[1].get("are_filled", None)
        pos_intervals = results[0].get("intervals", None)
        neg_intervals = results[1].get("intervals", None)

        # Get y-axis range for proper positioning
        y_min, y_max = ax.get_ylim()

        # Plot positive interval
        if pos_intervals is not None and len(pos_intervals) > 0:
            pos_interval_indices = pos_detections[pos_intervals]
            for i, (start_idx, end_idx) in enumerate(pos_interval_indices):
                start_time = time[start_idx]
                end_time = time[end_idx]

                ax.axvspan(
                    start_time,
                    end_time,
                    ymin=0.5,
                    ymax=1.0,
                    alpha=0.12,
                    color=color_peak,
                    zorder=1,
                    label="Positive Phase Cycle" if i == 0 else "",
                )

        # Plot negative intervals
        if neg_intervals is not None and len(neg_intervals) > 0:
            neg_interval_indices = neg_detections[neg_intervals]
            for i, (start_idx, end_idx) in enumerate(neg_interval_indices):
                start_time = time[start_idx]
                end_time = time[end_idx]

                ax.axvspan(
                    start_time,
                    end_time,
                    ymin=0.0,
                    ymax=0.5,
                    alpha=0.12,
                    color=color_trough,
                    zorder=1,
                    label="Negative Phase Cycle" if i == 0 else "",
                )

        # Draw midline separator
        ax.axhline(
            y=(y_min + y_max) / 2,
            color=color_grid,
            linestyle="--",
            linewidth=1.5,
            alpha=0.4,
            zorder=2,
        )

        # Plot positive detections
        if len(pos_detections) > 0:
            if pos_are_filled is not None and np.any(pos_are_filled):
                filled_pos = pos_detections[pos_are_filled]
                original_pos = pos_detections[~pos_are_filled]

                # Filled positive detections
                if len(filled_pos) > 0:
                    ax.scatter(
                        time[filled_pos],
                        signal_data[filled_pos],
                        color=color_filled,
                        s=180,
                        zorder=6,
                        marker="X",
                        edgecolors="white",
                        linewidths=2.5,
                        alpha=0.9,
                        label=f"Filled Peaks ({len(filled_pos)})",
                    )

                # Original positive detections
                if len(original_pos) > 0:
                    ax.scatter(
                        time[original_pos],
                        signal_data[original_pos],
                        color=color_peak,
                        s=120,
                        zorder=5,
                        marker="o",
                        edgecolors="white",
                        linewidths=2.5,
                        label=f"Positive Phase Peaks ({len(original_pos)})",
                    )
            else:
                # All positive detections
                ax.scatter(
                    time[pos_detections],
                    signal_data[pos_detections],
                    color=color_peak,
                    s=120,
                    zorder=5,
                    marker="o",
                    edgecolors="white",
                    linewidths=2.5,
                    label=f"Positive Phase Peaks ({len(pos_detections)})",
                )

        # Plot negative detections
        if len(neg_detections) > 0:
            if neg_are_filled is not None and np.any(neg_are_filled):
                filled_neg = neg_detections[neg_are_filled]
                original_neg = neg_detections[~neg_are_filled]

                # Filled negative detections
                if len(filled_neg) > 0:
                    ax.scatter(
                        time[filled_neg],
                        signal_data[filled_neg],
                        color=color_filled,
                        s=180,
                        zorder=6,
                        marker="X",
                        edgecolors="white",
                        linewidths=2.5,
                        alpha=0.9,
                        label=f"Filled Troughs ({len(filled_neg)})",
                    )

                # Original negative detections
                if len(original_neg) > 0:
                    ax.scatter(
                        time[original_neg],
                        signal_data[original_neg],
                        color=color_trough,
                        s=120,
                        zorder=5,
                        marker="o",
                        edgecolors="white",
                        linewidths=2.5,
                        label=f"Negative Phase Troughs ({len(original_neg)})",
                    )
            else:
                # All negative detections
                ax.scatter(
                    time[neg_detections],
                    signal_data[neg_detections],
                    color=color_trough,
                    s=120,
                    zorder=5,
                    marker="o",
                    edgecolors="white",
                    linewidths=2.5,
                    label=f"Negative Phase Troughs ({len(neg_detections)})",
                )

        # Calculate and display statistics
        total_cycles = min(len(pos_detections), len(neg_detections))
        if total_cycles > 1:
            duration_min = time[-1] / 60
            event_rate = total_cycles / duration_min if duration_min > 0 else 0

            stats_text = (
                f"Total Cycles: {total_cycles}\n"
                f"Rate: {event_rate:.1f} events/min\n"
                f"Duration: {time[-1]:.1f}s"
            )
            ax.text(
                0.0,
                -0.1,
                stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(
                    boxstyle="round,pad=0.6",
                    facecolor="white",
                    edgecolor=color_text,
                    linewidth=2,
                    alpha=0.95,
                ),
                color=color_text,
                fontfamily="monospace",
                fontweight="bold",
            )

        # Enhanced styling
        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold", color=color_text)
        ax.set_ylabel("Amplitude", fontsize=12, fontweight="bold", color=color_text)
        ax.set_title(
            "Detection - Positive & Negative Phases",
            fontsize=15,
            fontweight="bold",
            color=color_text,
            pad=15,
        )

        legend = ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.0, -0.05),
            fontsize=10,
            framealpha=0.95,
            edgecolor=color_text,
            fancybox=True,
            shadow=False,
            borderpad=1,
            labelspacing=0.8,
        )
        legend.get_frame().set_linewidth(2)

        # Grid and spines
        ax.grid(
            True, alpha=0.2, linestyle="--", linewidth=1, color=color_grid, zorder=0
        )
        ax.set_xlim(0, time[-1])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(color_text)
        ax.spines["bottom"].set_color(color_text)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.tick_params(colors=color_text, which="both", labelsize=10)

        plt.tight_layout(rect=[0, 0.02, 1, 1])
        plt.show()
        return

    # Case 3: Single phase (single subplot)
    fig, ax = plt.subplots(figsize=(20, 7))
    fig.patch.set_facecolor("white")

    _plot_single_phase_on_axis(
        ax=ax,
        signal_data=signal_data,
        results=results,
        time=time,
        color_signal=color_signal,
        color_peak=color_peak,
        color_filled=color_filled,
        color_grid=color_grid,
        color_text=color_text,
    )

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.show()
