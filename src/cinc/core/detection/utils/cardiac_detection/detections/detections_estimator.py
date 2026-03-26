"""
A class for estimating candidate cardiac detections from signal data.

This module provides the DetectionsEstimator class which identifies candidate
cardiac events (peaks or troughs) in physiological signals using prominence-based
peak detection algorithms.

Authors: Arnaud Poletto
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from typing import Tuple, Dict


class DetectionsEstimator:
    """
    A class for detecting candidatecardiac events in physiological signals using prominence-based peak detection.
    """

    def __init__(self, config: Dict, debug_plot: bool = False) -> None:
        """
        Initialize the detections estimator.

        Args:
            config (Dict): The configuration dictionary containing detection parameters.
                Must include 'min_separation_s', 'prominence_std_factor', and
                'prominence_wlen_s' keys.
            debug_plot (bool, optional): Whether to display debug plots showing
                detected peaks and troughs. Defaults to False.

        Raises:
            ValueError: If 'min_separation_s' is missing from config.
            ValueError: If 'prominence_std_factor' is missing from config.
            ValueError: If 'prominence_wlen_s' is missing from config.
        """
        if "min_separation_s" not in config:
            raise ValueError(
                "❌ 'min_separation_s' must be specified in the config dictionary."
            )
        if "prominence_std_factor" not in config:
            raise ValueError(
                "❌ 'prominence_std_factor' must be specified in the config dictionary."
            )
        if "prominence_wlen_s" not in config:
            raise ValueError(
                "❌ 'prominence_wlen_s' must be specified in the config dictionary."
            )

        super(DetectionsEstimator, self).__init__()

        self.debug_plot = debug_plot
        self.min_separation_s = config["min_separation_s"]
        self.prominence_std_factor = config["prominence_std_factor"]
        self.prominence_wlen_s = config["prominence_wlen_s"]

    def run(
        self,
        signal_data: np.ndarray,
        fs: float,
    ) -> Tuple[np.ndarray, int]:
        """
        Detect candidate cardiac events in the input signal.

        This method identifies both positive peaks and negative troughs in the signal,
        then selects the detection type (positive or negative) based on median prominence.
        The selected detections correspond to the more prominent cardiac events.

        Args:
            signal_data (np.ndarray): The input signal array with shape (n_samples,).
            fs (float): The sampling frequency in Hz.

        Returns:
            detections (np.ndarray): The array of detection indices in the signal.
            sign (int): The sign of selected detections (1 for positive, -1 for negative).
        """
        # Find peaks and troughs
        distance = int(self.min_separation_s * fs)
        prominence = np.std(signal_data) * self.prominence_std_factor
        prominence_wlen = int(self.prominence_wlen_s * fs)
        pos_detections, pos_properties = find_peaks(
            signal_data,
            distance=distance,
            prominence=prominence,
            wlen=prominence_wlen,
        )
        pos_prominences = pos_properties["prominences"]

        neg_detections, neg_properties = find_peaks(
            -signal_data,
            distance=distance,
            prominence=prominence,
            wlen=prominence_wlen,
        )
        neg_prominences = neg_properties["prominences"]

        # Handle case with no detections
        if len(pos_detections) == 0 and len(neg_detections) == 0:
            return np.array([]), 1

        # Handle cases where only one type exists
        if neg_detections.shape[0] == 0:
            selected_detections = pos_detections
            sign = 1
            detection_type = "positive"
        elif pos_detections.shape[0] == 0:
            selected_detections = neg_detections
            sign = -1
            detection_type = "negative"
        else:
            # Choose the type with higher prominence
            median_pos_prominence = np.median(pos_prominences)
            median_neg_prominence = np.median(neg_prominences)

            if median_pos_prominence > median_neg_prominence:
                selected_detections = pos_detections
                sign = 1
                detection_type = "positive"
            else:
                selected_detections = neg_detections
                sign = -1
                detection_type = "negative"

        # Plot detections if debug enabled
        if self.debug_plot:
            self._plot_detections(
                signal_data=signal_data,
                pos_detections=pos_detections,
                neg_detections=neg_detections,
                sign=sign,
                detection_type=detection_type,
            )

        return selected_detections, sign

    def _plot_detections(
        self,
        signal_data: np.ndarray,
        pos_detections: np.ndarray,
        neg_detections: np.ndarray,
        sign: int,
        detection_type: str,
    ) -> None:
        """
        Visualize detection results with professional styling.

        Creates a comprehensive plot showing both positive and negative phase detections,
        with visual emphasis on the phase selected by prominence comparison. The chosen
        phase is highlighted in green with larger markers, while the non-chosen phase
        appears in gray with smaller markers.

        Args:
            signal_data (np.ndarray): The input signal data with shape (n_samples,).
            pos_detections (np.ndarray): Indices of positive phase detections (peaks).
            neg_detections (np.ndarray): Indices of negative phase detections (troughs).
            sign (int): Selected sign (1 for positive, -1 for negative).
            detection_type (str): String describing selected type ("positive" or "negative").
        """
        # Define color palette
        color_selected = "#2ecc71"
        color_not_selected = "#95a5a6"
        color_signal = "#7f8c8d"
        color_text = "#34495e"
        color_grid = "#bdc3c7"

        fig, ax = plt.subplots(figsize=(15, 6), dpi=100)
        fig.patch.set_facecolor("white")

        # Plot signal
        sample_indices = np.arange(len(signal_data))
        ax.plot(
            sample_indices,
            signal_data,
            color=color_signal,
            linewidth=2,
            alpha=0.6,
            label="Signal",
            zorder=2,
        )

        # Configure visual properties for chosen vs non-chosen phases
        pos_color = color_selected if sign == 1 else color_not_selected
        neg_color = color_selected if sign == -1 else color_not_selected

        pos_size = 120 if sign == 1 else 80
        neg_size = 120 if sign == -1 else 80

        pos_zorder = 5 if sign == 1 else 3
        neg_zorder = 5 if sign == -1 else 3

        pos_linewidth = 2.5 if sign == 1 else 1.5
        neg_linewidth = 2.5 if sign == -1 else 1.5

        # Plot negative phase detections
        if len(neg_detections) > 0:
            neg_label = (
                f"Chosen Phase ({len(neg_detections)})"
                if sign == -1
                else f"Not Chosen ({len(neg_detections)})"
            )
            ax.scatter(
                neg_detections,
                signal_data[neg_detections],
                color=neg_color,
                s=neg_size,
                marker="o",
                edgecolors="white",
                linewidths=neg_linewidth,
                label=neg_label,
                zorder=neg_zorder,
            )

        # Plot positive phase detections
        if len(pos_detections) > 0:
            pos_label = (
                f"Chosen Phase ({len(pos_detections)})"
                if sign == 1
                else f"Not Chosen ({len(pos_detections)})"
            )
            ax.scatter(
                pos_detections,
                signal_data[pos_detections],
                color=pos_color,
                s=pos_size,
                marker="o",
                edgecolors="white",
                linewidths=pos_linewidth,
                label=pos_label,
                zorder=pos_zorder,
            )

        # Display selection summary in top-left badge
        badge_text = (
            f"Chosen Phase: {detection_type.capitalize()}\n"
            f"Positive Prominence: {len(pos_detections)}\n"
            f"Negative Prominence: {len(neg_detections)}"
        )
        badge_color = color_selected
        ax.text(
            0.02,
            0.97,
            badge_text,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=badge_color,
                edgecolor="white",
                linewidth=2,
                alpha=0.9,
            ),
            color="white",
            zorder=10,
        )
        ax.set_xlabel("Sample Index", fontsize=12, fontweight="bold", color=color_text)
        ax.set_ylabel("Amplitude", fontsize=12, fontweight="bold", color=color_text)

        # Set title
        title_text = (
            f"Cardiac Detection Selection - {detection_type.capitalize()} Phase Chosen"
        )
        ax.set_title(
            title_text,
            fontsize=15,
            fontweight="bold",
            color=color_text,
            pad=15,
        )

        # Configure legend
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

        # Configure grid and axes
        ax.grid(
            True, alpha=0.2, linestyle="--", linewidth=1, color=color_grid, zorder=0
        )
        ax.set_xlim(0, len(signal_data))

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(color_text)
        ax.spines["bottom"].set_color(color_text)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.tick_params(colors=color_text, which="both", labelsize=10)

        plt.tight_layout(rect=[0, 0.02, 1, 1])
        plt.show()
