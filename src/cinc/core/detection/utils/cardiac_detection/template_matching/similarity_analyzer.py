"""
Similarity analyzer for cardiac detection using amplitude-corrected correlation.

This module provides the SimilarityAnalyzer class which computes similarity
scores between detected peaks and a reference shape template using
amplitude-corrected cross-correlation with optimal shift detection.

Authors: Arnaud Poletto
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Any, List, Tuple, Dict

from cinc.core.detection.utils.cardiac_detection.amplitude_corrected_correlation import (
    amplitude_corrected_correlate,
)


class SimilarityAnalyzer:
    """
    A class for analyzing similarity between detected peaks and a reference template shape.
    """

    def __init__(
        self, config: Dict, debug_print: bool = False, debug_plot: bool = False
    ) -> None:
        """
        Initialize the similarity analyzer with configuration parameters.

        Args:
            config (Dict): The configuration dictionary with required parameters.
            debug_print (bool, optional): Whether to enable debug printing. Defaults to False.
            debug_plot (bool, optional): Whether to enable debug plotting. Defaults to False.

        Raises:
            ValueError: If 'max_lag_s' is missing from config.
            ValueError: If 'correlation_penalty_range' is missing from config.
            ValueError: If 'correlation_penalty_quantile_percent' is missing from config.
        """
        if "max_lag_s" not in config:
            raise ValueError(
                "❌ 'max_lag_s' must be specified in the config dictionary."
            )
        if "correlation_penalty_range" not in config:
            raise ValueError(
                "❌ 'correlation_penalty_range' must be specified in the config dictionary."
            )
        if "correlation_penalty_quantile_percent" not in config:
            raise ValueError(
                "❌ 'correlation_penalty_quantile_percent' must be specified in the config dictionary."
            )

        super(SimilarityAnalyzer, self).__init__()

        self.debug_print = debug_print
        self.debug_plot = debug_plot
        self.max_lag_s = config["max_lag_s"]
        self.correlation_penalty_range = config["correlation_penalty_range"]
        self.correlation_penalty_quantile_percent = config[
            "correlation_penalty_quantile_percent"
        ]

    def _find_optimal_shift(
        self,
        search_correlation: np.ndarray,
        start_lag: int,
        center: int,
        norm_reference: float,
        norm_segment: float,
    ) -> Tuple[float, int]:
        """
        Find the optimal shift that maximizes the normalized correlation score.

        This method searches through the correlation array to find the lag that
        produces the highest normalized correlation between the reference shape
        and the signal segment.

        Args:
            search_correlation (np.ndarray): The correlation values across lags.
            start_lag (int): The starting lag index in the full correlation array.
            center (int): The center index of the full correlation array.
            norm_reference (float): The L2 norm of the reference shape.
            norm_segment (float): The L2 norm of the signal segment.

        Returns:
            best_score (float): The highest normalized correlation score found.
            optimal_shift (int): The shift amount in samples that maximizes correlation.
        """
        if len(search_correlation) == 0 or norm_reference == 0 or norm_segment == 0:
            return 0.0, 0

        best_score = -1.0
        optimal_shift = 0
        for lag_idx in range(len(search_correlation)):
            shift = lag_idx + start_lag - center

            # Normalize the amplitude-corrected correlation
            raw_corrected_correlation = search_correlation[lag_idx]
            normalized_corrected_correlation = raw_corrected_correlation / (
                norm_reference * norm_segment
            )

            # Check if this is the best score so far
            if normalized_corrected_correlation > best_score:
                best_score = normalized_corrected_correlation
                optimal_shift = shift

        return best_score, optimal_shift

    def run(
        self,
        upsampled_signal_data: np.ndarray,
        upsampled_detections: np.ndarray,
        upsampled_prominences: np.ndarray,
        upsampled_fs: float,
        upsampled_reference_shape: np.ndarray,
        shape_offset: int,
    ) -> Dict:
        """
        Run similarity analysis on detected peaks using the reference shape.

        This method computes amplitude-corrected cross-correlation between each
        detected peak's surrounding segment and the reference shape template,
        finding the optimal shift and correlation score for each valid detection.

        Args:
            upsampled_signal_data (np.ndarray): The upsampled signal data array.
            upsampled_detections (np.ndarray): The array of detection indices.
            upsampled_prominences (np.ndarray): The array of prominence values.
            upsampled_fs (float): The upsampled sampling frequency in Hz.
            upsampled_reference_shape (np.ndarray): The reference shape template array.
            shape_offset (int): The offset in samples from detection to shape window start.

        Returns:
            similarity_results (Dict): A dictionary containing detected peaks, their prominences,
            optimal shifts, and correlation scores.
        """
        # Initialize result arrays
        upsampled_reference_shape_length = upsampled_reference_shape.shape[0]
        upsampled_signal_data_length = upsampled_signal_data.shape[0]
        valid_upsampled_detections = []
        valid_upsampled_prominences = []
        shifts = []
        correlations = []
        for detection, prominence in zip(upsampled_detections, upsampled_prominences):
            # Define window for current segment
            w_start = detection + shape_offset
            w_end = w_start + upsampled_reference_shape_length
            if w_start < 0 or w_end >= upsampled_signal_data_length:
                continue

            # Get current segment from signal data
            current_segment = upsampled_signal_data[w_start:w_end]
            if len(current_segment) != upsampled_reference_shape_length:
                continue

            # Compute amplitude-corrected cross-correlation across all possible lags
            correlation = amplitude_corrected_correlate(
                upsampled_reference_shape,
                current_segment,
                mode="full",
                penalty_range=self.correlation_penalty_range,
                quantile_percent=self.correlation_penalty_quantile_percent,
            )
            center = len(correlation) // 2
            upsampled_max_lag = int(self.max_lag_s * upsampled_fs)
            start_lag = max(0, center - upsampled_max_lag)
            end_lag = min(len(correlation), center + upsampled_max_lag + 1)
            search_correlation = correlation[start_lag:end_lag]

            # Calculate norms
            norm_reference = np.linalg.norm(upsampled_reference_shape)
            norm_segment = np.linalg.norm(current_segment)
            if norm_reference == 0 or norm_segment == 0:
                continue

            # Find optimal shift and compute correlation
            normalized_corrected_correlation, optimal_shift = self._find_optimal_shift(
                search_correlation=search_correlation,
                start_lag=start_lag,
                center=center,
                norm_reference=norm_reference,
                norm_segment=norm_segment,
            )

            # Store results
            valid_upsampled_detections.append(detection)
            valid_upsampled_prominences.append(prominence)
            shifts.append(optimal_shift)
            correlations.append(normalized_corrected_correlation)

        # Plot analysis if debug enabled
        if self.debug_plot:
            self._plot_similarity_analysis(
                upsampled_signal_data=upsampled_signal_data,
                upsampled_detections=valid_upsampled_detections,
                upsampled_reference_shape=upsampled_reference_shape,
                shape_offset=shape_offset,
                correlations=correlations,
                shifts=shifts,
            )

        detections = np.array(valid_upsampled_detections, dtype=int)
        prominences = np.array(valid_upsampled_prominences, dtype=float)
        shifts = np.array(shifts, dtype=int)
        correlations = np.array(correlations, dtype=float)

        return {
            "detections": detections,
            "prominences": prominences,
            "shifts": shifts,
            "correlations": correlations,
        }

    def _plot_similarity_analysis(
        self,
        upsampled_signal_data: np.ndarray,
        upsampled_detections: List[int],
        upsampled_reference_shape: np.ndarray,
        shape_offset: int,
        correlations: List[float],
        shifts: List[int],
        cmap: Any = plt.cm.RdYlGn,
    ) -> None:
        """
        Plot the similarity analysis results with template overlays and metrics.

        This method creates a comprehensive visualization showing the signal with
        template overlays, the reference shape, correlation quality, and shift
        analysis across all detections.

        Args:
            upsampled_signal_data (np.ndarray): The upsampled signal data array.
            upsampled_detections (List[int]): The list of detection indices.
            upsampled_reference_shape (np.ndarray): The reference shape template array.
            shape_offset (int): The offset in samples from detection to shape window start.
            correlations (List[float]): The list of correlation scores.
            shifts (List[int]): The list of shift amounts in samples.
            cmap (Any): The colormap for visualization. Defaults to plt.cm.RdYlGn.
        """
        if len(upsampled_detections) == 0:
            print("⚠️  No detections to plot in similarity analysis.")
            return

        # Define color palette
        color_signal = "#7f8c8d"
        color_text = "#34495e"
        color_grid = "#bdc3c7"

        # Create figure with GridSpec layout
        fig = plt.figure(figsize=(18, 12), dpi=100)
        fig.patch.set_facecolor("white")
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

        # Plot 1: Signal with template overlays (top row, spans full width)
        ax1 = fig.add_subplot(gs[0, :])
        sample_indices = np.arange(len(upsampled_signal_data))
        ax1.plot(
            sample_indices,
            upsampled_signal_data,
            color=color_signal,
            linewidth=2,
            alpha=0.6,
            label="Signal",
            zorder=2,
        )

        # Plot template overlays at detection locations
        for i, (detection, shift, correlation) in enumerate(
            zip(upsampled_detections, shifts, correlations)
        ):
            # Calculate the correct template position
            template_start = detection + shape_offset - shift
            template_end = template_start + len(upsampled_reference_shape)
            rounded_template_start = int(np.round(template_start))
            rounded_template_end = int(np.round(template_end))

            if (
                0
                <= rounded_template_start
                < rounded_template_end
                <= len(upsampled_signal_data)
            ):
                template_indices = np.arange(
                    rounded_template_start, rounded_template_end
                )
                # Scale the reference to match the signal amplitude at this location
                signal_segment = upsampled_signal_data[
                    rounded_template_start:rounded_template_end
                ]
                template_amplitude = np.max(np.abs(upsampled_reference_shape))
                signal_amplitude = np.max(np.abs(signal_segment))
                if template_amplitude > 0:
                    scale_factor = signal_amplitude / template_amplitude
                    template_scaled = upsampled_reference_shape * scale_factor
                    # Offset to match the signal baseline
                    template_offset = np.mean(signal_segment) - np.mean(template_scaled)
                    template_scaled += template_offset
                else:
                    template_scaled = upsampled_reference_shape

                color = cmap(np.clip(correlation, 0, 1))
                ax1.plot(
                    template_indices,
                    template_scaled,
                    color=color,
                    alpha=0.25,
                    linewidth=3,
                    label="Template Overlay" if i == 0 else None,
                    zorder=3,
                )

        # Mark detection points
        if len(upsampled_detections) > 0:
            colors = cmap(np.clip(correlations, 0, 1))
            ax1.scatter(
                upsampled_detections,
                upsampled_signal_data[upsampled_detections],
                c=colors,
                s=120,
                marker="o",
                edgecolors="white",
                linewidths=2,
                label=f"Detections ({len(upsampled_detections)})",
                zorder=5,
            )

        # Configure labels and title
        ax1.set_xlabel("Sample Index", fontsize=12, fontweight="bold", color=color_text)
        ax1.set_ylabel("Amplitude", fontsize=12, fontweight="bold", color=color_text)
        ax1.set_title(
            "Template Matching - Signal with Overlays",
            fontsize=15,
            fontweight="bold",
            color=color_text,
            pad=15,
        )

        # Configure legend (outside plot, top right)
        legend = ax1.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
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
        ax1.grid(
            True, alpha=0.2, linestyle="--", linewidth=1, color=color_grid, zorder=0
        )
        ax1.set_xlim(0, len(upsampled_signal_data))
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["left"].set_color(color_text)
        ax1.spines["bottom"].set_color(color_text)
        ax1.spines["left"].set_linewidth(1.5)
        ax1.spines["bottom"].set_linewidth(1.5)
        ax1.tick_params(colors=color_text, which="both", labelsize=10)

        # Plot 2: Reference template shape (bottom left)
        ax2 = fig.add_subplot(gs[1, 0])
        shape_indices = np.arange(len(upsampled_reference_shape))
        ax2.plot(
            shape_indices,
            upsampled_reference_shape,
            color="#9b59b6",
            linewidth=3,
            label="Reference Template",
            zorder=2,
        )

        ax2.set_xlabel(
            "Sample Index (relative)", fontsize=12, fontweight="bold", color=color_text
        )
        ax2.set_ylabel("Amplitude", fontsize=12, fontweight="bold", color=color_text)
        ax2.set_title(
            "Reference Template Shape",
            fontsize=15,
            fontweight="bold",
            color=color_text,
            pad=15,
        )

        legend2 = ax2.legend(
            loc="upper right",
            fontsize=10,
            framealpha=0.95,
            edgecolor=color_text,
            fancybox=True,
            shadow=False,
            borderpad=1,
        )
        legend2.get_frame().set_linewidth(2)

        ax2.grid(
            True, alpha=0.2, linestyle="--", linewidth=1, color=color_grid, zorder=0
        )
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_color(color_text)
        ax2.spines["bottom"].set_color(color_text)
        ax2.spines["left"].set_linewidth(1.5)
        ax2.spines["bottom"].set_linewidth(1.5)
        ax2.tick_params(colors=color_text, which="both", labelsize=10)

        # Plot 3: Correlation quality distribution (bottom right)
        ax3 = fig.add_subplot(gs[1, 1])

        colors_scatter = cmap(np.clip(correlations, 0, 1))
        if len(upsampled_detections) > 0:
            ax3.scatter(
                correlations,
                np.random.rand(len(upsampled_detections)),
                c=colors_scatter,
                s=100,
                marker="o",
                edgecolors="white",
                linewidths=2,
                zorder=3,
            )

        ax3.set_xlabel(
            "Correlation Coefficient", fontsize=12, fontweight="bold", color=color_text
        )
        ax3.set_title(
            "Quality Assessment Distribution",
            fontsize=15,
            fontweight="bold",
            color=color_text,
            pad=15,
        )

        ax3.grid(
            True, alpha=0.2, linestyle="--", linewidth=1, color=color_grid, zorder=0
        )
        ax3.set_yticks([])
        ax3.set_xlim(-0.1, 1.05)
        ax3.set_ylim(-0.05, 1.05)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        ax3.spines["left"].set_color(color_text)
        ax3.spines["bottom"].set_color(color_text)
        ax3.spines["left"].set_linewidth(1.5)
        ax3.spines["bottom"].set_linewidth(1.5)
        ax3.tick_params(colors=color_text, which="both", labelsize=10)

        # Plot 4: Shift analysis (bottom row, spans all columns)
        ax4 = fig.add_subplot(gs[2, :])

        if len(upsampled_detections) > 0:
            # Create stem plot for shifts
            for i, (pos, shift) in enumerate(zip(upsampled_detections, shifts)):
                correlation = correlations[i]
                color = cmap(np.clip(correlation, 0, 1))
                ax4.scatter(
                    pos,
                    shift,
                    color=color,
                    s=100,
                    marker="o",
                    edgecolors="white",
                    linewidths=2,
                    zorder=3,
                )
                ax4.plot(
                    [pos, pos],
                    [0, shift],
                    color=color,
                    alpha=0.3,
                    linewidth=2,
                    zorder=2,
                )

            # Add zero reference line
            ax4.axhline(y=0, color=color_text, linestyle="-", linewidth=1.5, alpha=0.5)

            ax4.set_xlabel(
                "Sample Index", fontsize=12, fontweight="bold", color=color_text
            )
            ax4.set_ylabel(
                "Shift Amount (samples)",
                fontsize=12,
                fontweight="bold",
                color=color_text,
            )
            ax4.set_title(
                "Alignment Shifts",
                fontsize=15,
                fontweight="bold",
                color=color_text,
                pad=15,
            )

            ax4.grid(
                True, alpha=0.2, linestyle="--", linewidth=1, color=color_grid, zorder=0
            )
            ax4.set_xlim(0, len(upsampled_signal_data))
            ax4.spines["top"].set_visible(False)
            ax4.spines["right"].set_visible(False)
            ax4.spines["left"].set_color(color_text)
            ax4.spines["bottom"].set_color(color_text)
            ax4.spines["left"].set_linewidth(1.5)
            ax4.spines["bottom"].set_linewidth(1.5)
            ax4.tick_params(colors=color_text, which="both", labelsize=10)

        # Overall title
        plt.suptitle(
            f"Similarity Analysis - {len(upsampled_detections)} Detections",
            fontsize=16,
            fontweight="bold",
            color=color_text,
            y=0.98,
        )

        plt.show()
