"""
A class for clustering cardiac detections based on shape similarity.

This module implements the SimilarityClusterer class which groups cardiac detections
by comparing their waveform shapes using amplitude-corrected cross-correlation.
It iteratively creates clusters by selecting reference patterns and assigning
similar detections to the same cluster.

Authors: Arnaud Poletto
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional

from cinc.core.detection.utils.cardiac_detection.amplitude_corrected_correlation import (
    amplitude_corrected_correlate,
)


class SimilarityClusterer:
    """A class for clustering cardiac detections based on shape similarity using correlation analysis."""

    def __init__(
        self,
        config: Dict,
        debug_print: bool = False,
        debug_plot: bool = False,
    ) -> None:
        """
        Initialize the similarity clusterer with configuration and debug settings.

        Args:
            config (Dict): The configuration dictionary with required parameters.
            debug_print (bool, optional): Whether to enable debug printing. Defaults to False.
            debug_plot (bool, optional): Whether to enable debug plotting. Defaults to False.

        Raises:
            ValueError: If 'correlation_penalty_range' is missing from config.
            ValueError: If 'correlation_penalty_quantile_percent' is missing from config.
            ValueError: If 'max_lag_s' is missing from config.
            ValueError: If 'good_correlation_threshold' is missing from config.
        """
        if "correlation_penalty_range" not in config:
            raise ValueError(
                "❌ 'correlation_penalty_range' must be specified in the config dictionary."
            )
        if "correlation_penalty_quantile_percent" not in config:
            raise ValueError(
                "❌ 'correlation_penalty_quantile_percent' must be specified in the config dictionary."
            )
        if "max_lag_s" not in config:
            raise ValueError(
                "❌ 'max_lag_s' must be specified in the config dictionary."
            )
        if "good_correlation_threshold" not in config:
            raise ValueError(
                "❌ 'good_correlation_threshold' must be specified in the config dictionary."
            )

        super(SimilarityClusterer, self).__init__()

        self.debug_print = debug_print
        self.debug_plot = debug_plot
        self.correlation_penalty_range = config["correlation_penalty_range"]
        self.correlation_penalty_quantile_percent = config[
            "correlation_penalty_quantile_percent"
        ]
        self.max_lag_s = config["max_lag_s"]
        self.good_correlation_threshold = config["good_correlation_threshold"]

    def _find_first_valid_reference(
        self,
        signal_data: np.ndarray,
        detections: np.ndarray,
        shape_window: Tuple[int, int],
        cluster_labels: np.ndarray,
    ) -> Optional[int]:
        """
        Find the first valid reference detection within signal boundaries.

        Args:
            signal_data (np.ndarray): The input signal data.
            detections (np.ndarray): The array of detection indices.
            shape_window (Tuple[int, int]): The window around each detection (start_offset, end_offset).
            cluster_labels (np.ndarray): The array of cluster assignments (modified in-place).

        Returns:
            reference_detection (Optional[int]): The index of the first valid reference detection, or None if none found.
        """
        for reference_detection in range(len(detections)):
            idx = detections[reference_detection]
            w_start = idx + shape_window[0]
            w_end = idx + shape_window[1]

            if 0 <= w_start < w_end < len(signal_data):
                return reference_detection
            else:
                # Mark invalid detections
                cluster_labels[reference_detection] = 0

        return None

    def _find_optimal_shift(
        self,
        search_correlation: np.ndarray,
        start_lag: int,
        center: int,
        norm_reference: float,
        norm_segment: float,
    ) -> Tuple[float, int]:
        """
        Find the optimal shift that maximizes normalized correlation.

        Args:
            search_correlation (np.ndarray): The correlation values within search range.
            start_lag (int): The starting lag index.
            center (int): The center index of the full correlation.
            norm_reference (float): The norm of the reference pattern.
            norm_segment (float): The norm of the current segment.

        Returns:
            best_score (float): The best normalized correlation coefficient.
            optimal_shift (int): The optimal shift value in samples.
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

    def _assign_detections_to_cluster(
        self,
        signal_data: np.ndarray,
        detections: np.ndarray,
        fs: float,
        reference_pattern: np.ndarray,
        shape_window: Tuple[int, int],
        cluster_labels: np.ndarray,
        shifts: np.ndarray,
        cluster_id: int,
    ) -> None:
        """
        Assign detections to a cluster based on correlation with reference pattern.

        Args:
            signal_data (np.ndarray): The input signal data.
            detections (np.ndarray): The array of detection indices.
            fs (float): The sampling frequency in Hz.
            reference_pattern (np.ndarray): The reference pattern for this cluster.
            shape_window (Tuple[int, int]): The window around each detection (start_offset, end_offset).
            cluster_labels (np.ndarray): The array of cluster assignments (modified in-place).
            shifts (np.ndarray): The array of optimal shifts for each detection (modified in-place).
            cluster_id (int): The ID of the current cluster.
        """
        for i, detection in enumerate(detections):
            if not np.isnan(cluster_labels[i]):
                continue

            # Check boundary conditions
            window_start = detection + shape_window[0]
            window_end = detection + shape_window[1]
            if window_start < 0 or window_end >= len(signal_data):
                cluster_labels[i] = 0  # Mark as invalid
                continue

            # Extract current pattern
            current_pattern = signal_data[window_start:window_end]

            # Compute amplitude-corrected cross-correlation
            penalty_range = self.correlation_penalty_range
            quantile_percent = self.correlation_penalty_quantile_percent
            correlation = amplitude_corrected_correlate(
                current_pattern,
                reference_pattern,
                mode="full",
                penalty_range=penalty_range,
                quantile_percent=quantile_percent,
            )
            center = len(correlation) // 2
            max_lag = int(self.max_lag_s * fs)
            start_lag = max(0, center - max_lag)
            end_lag = min(len(correlation), center + max_lag + 1)
            search_correlation = correlation[start_lag:end_lag]

            # Calculate norms
            norm_reference = np.linalg.norm(reference_pattern)
            norm_current = np.linalg.norm(current_pattern)
            if norm_reference == 0 or norm_current == 0:
                continue

            # Compute correlation and find best alignment
            correlation_coeff, optimal_shift = self._find_optimal_shift(
                search_correlation=search_correlation,
                start_lag=start_lag,
                center=center,
                norm_reference=norm_reference,
                norm_segment=norm_current,
            )

            # Assign to cluster if correlation meets threshold
            if correlation_coeff >= self.good_correlation_threshold:
                cluster_labels[i] = cluster_id
                shifts[i] = optimal_shift

    def _remove_small_clusters(
        self,
        cluster_labels: np.ndarray,
        min_size: int,
    ) -> np.ndarray:
        """
        Remove clusters that have fewer detections than the minimum size.

        Args:
            cluster_labels (np.ndarray): The array of cluster assignments.
            min_size (int): The minimum number of detections required for a valid cluster.

        Returns:
            cluster_labels (np.ndarray): The updated cluster labels array with small clusters removed.
        """
        unique_clusters, counts = np.unique(
            cluster_labels[~np.isnan(cluster_labels)], return_counts=True
        )

        # Remove small clusters
        small_clusters = unique_clusters[(counts < min_size) & (unique_clusters > 0)]
        for cluster_id in small_clusters:
            cluster_labels[cluster_labels == cluster_id] = np.nan

        if self.debug_print:
            n_removed = len(small_clusters)
            if n_removed > 0:
                print(f"⚠️  Removed {n_removed} small clusters (size < {min_size}).")

        return cluster_labels

    def run(
        self,
        signal_data: np.ndarray,
        detections: np.ndarray,
        fs: float,
        shape_window: Tuple[int, int],
    ) -> np.ndarray:
        """
        Cluster cardiac detections based on shape similarity using correlation.

        This method iteratively creates clusters by selecting reference patterns and
        assigning detections with high correlation to the same cluster. Small clusters
        are removed to ensure statistical validity.

        Args:
            signal_data (np.ndarray): The input signal data.
            detections (np.ndarray): The array of detection indices.
            fs (float): The sampling frequency in Hz.
            shape_window (Tuple[int, int]): The window around each detection (start_offset, end_offset).

        Returns:
            cluster_labels (np.ndarray): The cluster assignment for each detection.
            shifts (np.ndarray): The optimal temporal shift for each detection.
        """
        n_detections = detections.shape[0]
        cluster_labels = np.full(n_detections, np.nan)
        shifts = np.zeros(n_detections, dtype=float)
        cluster_id = 0

        # Find first valid reference detection
        reference_detection = self._find_first_valid_reference(
            signal_data=signal_data,
            detections=detections,
            shape_window=shape_window,
            cluster_labels=cluster_labels,
        )
        if reference_detection is None:
            if self.debug_print:
                print("⚠️  No valid detections found for clustering.")
            return cluster_labels, shifts

        # Create clusters until all detections are assigned
        while reference_detection is not None:
            cluster_id += 1
            ref_signal_idx = detections[reference_detection]

            # Extract reference pattern for this cluster
            reference_start = ref_signal_idx + shape_window[0]
            reference_end = ref_signal_idx + shape_window[1]
            reference_pattern = signal_data[reference_start:reference_end]

            # Assign detections to current cluster based on correlation
            self._assign_detections_to_cluster(
                signal_data=signal_data,
                detections=detections,
                fs=fs,
                reference_pattern=reference_pattern,
                shape_window=shape_window,
                cluster_labels=cluster_labels,
                shifts=shifts,
                cluster_id=cluster_id,
            )

            # Find next unassigned detection as reference for next cluster
            unassigned_detections = np.where(np.isnan(cluster_labels))[0]
            reference_detection = (
                unassigned_detections[0] if len(unassigned_detections) > 0 else None
            )

        # Remove small clusters
        cluster_labels = self._remove_small_clusters(cluster_labels, min_size=2)

        if self.debug_print:
            n_valid_clusters = len(
                np.unique(
                    cluster_labels[~np.isnan(cluster_labels) & (cluster_labels > 0)]
                )
            )
            print(f"✅ Clustering complete. Found {n_valid_clusters} valid clusters.")

        # Plot cluster analysis if debug enabled
        if self.debug_plot:
            self._plot_cluster_analysis(
                signal_data=signal_data,
                detections=detections,
                cluster_labels=cluster_labels,
                shape_window=shape_window,
            )

        return cluster_labels, shifts

    def _plot_cluster_analysis(
        self,
        signal_data: np.ndarray,
        detections: np.ndarray,
        cluster_labels: np.ndarray,
        shape_window: Tuple[int, int],
    ) -> None:
        """
        Visualize cluster analysis results with comprehensive plots.

        Creates a visualization showing signal with cluster assignments, cluster size
        distribution sorted from largest to smallest, and overlaid shape patterns
        for each cluster.

        Args:
            signal_data (np.ndarray): The input signal data.
            detections (np.ndarray): The array of detection indices.
            cluster_labels (np.ndarray): The cluster assignment for each detection.
            shape_window (Tuple[int, int]): The window for shape extraction (start_offset, end_offset).
        """
        # Get valid clusters
        valid_clusters = cluster_labels[~np.isnan(cluster_labels)]
        valid_clusters = valid_clusters[valid_clusters > 0]
        if len(valid_clusters) == 0:
            print("⚠️  No valid clusters to plot.")
            return

        unique_clusters, counts = np.unique(valid_clusters, return_counts=True)
        n_clusters = len(unique_clusters)

        # Sort clusters by size
        sorted_indices = np.argsort(counts)[::-1]
        unique_clusters = unique_clusters[sorted_indices]
        counts = counts[sorted_indices]

        # Define color palette
        color_signal = "#7f8c8d"
        color_text = "#34495e"
        color_grid = "#bdc3c7"
        color_invalid = "#e74c3c"

        # Create distinct colors for clusters
        colors = plt.cm.Set3(np.linspace(0, 1, max(n_clusters, 3)))

        # Plot 1: Signal with cluster assignments
        fig = plt.figure(figsize=(18, 12), dpi=100)
        fig.patch.set_facecolor("white")
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

        # Plot signal with cluster assignments
        ax1 = fig.add_subplot(gs[0, :2])
        sample_indices = np.arange(len(signal_data))
        ax1.plot(
            sample_indices,
            signal_data,
            color=color_signal,
            alpha=0.6,
            linewidth=1.5,
            label="Signal",
            zorder=1,
        )

        # Plot each cluster with different colors
        for i, cluster_id in enumerate(unique_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_detections = detections[cluster_mask]
            color = colors[i]

            # Only add label for the first cluster to create custom legend entry later
            ax1.scatter(
                cluster_detections,
                signal_data[cluster_detections],
                color=color,
                s=100,
                marker="o",
                edgecolors="white",
                linewidths=2,
                zorder=3,
            )

        # Plot invalid/unassigned detections
        invalid_mask = (cluster_labels == 0) | np.isnan(cluster_labels)
        if np.any(invalid_mask):
            invalid_detections = detections[invalid_mask]
            ax1.scatter(
                invalid_detections,
                signal_data[invalid_detections],
                color=color_invalid,
                s=60,
                alpha=0.7,
                marker="x",
                linewidths=2,
                label=f"Invalid ({np.sum(invalid_mask)})",
                zorder=2,
            )

        # Create compact legend with minimal entries
        # Add text annotation showing cluster info instead of cluttering legend
        cluster_info = f"{n_clusters} Clusters"
        ax1.text(
            0.02,
            0.97,
            cluster_info,
            transform=ax1.transAxes,
            fontsize=10,
            fontweight="bold",
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor=color_text,
                linewidth=1.5,
                alpha=0.95,
            ),
            color=color_text,
            zorder=10,
        )

        ax1.set_xlabel("Sample Index", fontsize=12, fontweight="bold", color=color_text)
        ax1.set_ylabel("Amplitude", fontsize=12, fontweight="bold", color=color_text)
        ax1.set_title(
            f"Signal with {n_clusters} Detected Clusters",
            fontsize=15,
            fontweight="bold",
            color=color_text,
            pad=15,
        )

        # Simple legend with just Signal and Invalid if present
        if np.any(invalid_mask):
            legend1 = ax1.legend(
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
            legend1.get_frame().set_linewidth(2)

        ax1.grid(
            True, alpha=0.2, linestyle="--", linewidth=1, color=color_grid, zorder=0
        )
        ax1.set_xlim(0, len(signal_data))
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["left"].set_color(color_text)
        ax1.spines["bottom"].set_color(color_text)
        ax1.spines["left"].set_linewidth(1.5)
        ax1.spines["bottom"].set_linewidth(1.5)
        ax1.tick_params(colors=color_text, which="both", labelsize=10)

        # Plot 2: Cluster size distribution
        ax2 = fig.add_subplot(gs[1, 0])
        x_positions = np.arange(len(unique_clusters))
        bars = ax2.bar(
            x_positions,
            counts,
            color=colors[: len(unique_clusters)],
            edgecolor=color_text,
            linewidth=1.5,
        )

        ax2.set_xlabel("Cluster ID", fontsize=12, fontweight="bold", color=color_text)
        ax2.set_ylabel(
            "Number of Detections", fontsize=12, fontweight="bold", color=color_text
        )
        ax2.set_title(
            "Cluster Sizes",
            fontsize=15,
            fontweight="bold",
            color=color_text,
            pad=15,
        )
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels([f"C{int(c)}" for c in unique_clusters])

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.02,
                str(count),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color=color_text,
            )

        ax2.grid(
            True,
            alpha=0.2,
            linestyle="--",
            linewidth=1,
            color=color_grid,
            zorder=0,
            axis="y",
        )
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_color(color_text)
        ax2.spines["bottom"].set_color(color_text)
        ax2.spines["left"].set_linewidth(1.5)
        ax2.spines["bottom"].set_linewidth(1.5)
        ax2.tick_params(colors=color_text, which="both", labelsize=10)

        # Plot 3: Overlaid shapes for each cluster
        ax3 = fig.add_subplot(gs[1, 1])
        cluster_averages = {}

        for i, cluster_id in enumerate(unique_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_detections = detections[cluster_mask]

            # Extract and align shapes for this cluster
            shapes = []
            for detection in cluster_detections:
                w_start = detection + shape_window[0]
                w_end = detection + shape_window[1]
                if 0 <= w_start < w_end < len(signal_data):
                    shape = signal_data[w_start:w_end]
                    shapes.append(shape)

            if len(shapes) > 0:
                # Plot individual shapes with transparency
                for shape in shapes:
                    ax3.plot(shape, color=colors[i], alpha=0.2, linewidth=0.8, zorder=1)

                # Calculate and plot average with emphasis
                avg_shape = np.mean(shapes, axis=0)
                cluster_averages[cluster_id] = avg_shape
                ax3.plot(
                    avg_shape,
                    color=colors[i],
                    linewidth=3,
                    zorder=2,
                )

        ax3.set_xlabel(
            "Sample Index (relative)", fontsize=12, fontweight="bold", color=color_text
        )
        ax3.set_ylabel("Amplitude", fontsize=12, fontweight="bold", color=color_text)
        ax3.set_title(
            "Cluster Shape Patterns",
            fontsize=15,
            fontweight="bold",
            color=color_text,
            pad=15,
        )

        ax3.grid(
            True, alpha=0.2, linestyle="--", linewidth=1, color=color_grid, zorder=0
        )
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        ax3.spines["left"].set_color(color_text)
        ax3.spines["bottom"].set_color(color_text)
        ax3.spines["left"].set_linewidth(1.5)
        ax3.spines["bottom"].set_linewidth(1.5)
        ax3.tick_params(colors=color_text, which="both", labelsize=10)

        # Overall title
        plt.suptitle(
            f"Cluster Analysis - {n_clusters} Clusters, {len(detections)} Total Detections",
            fontsize=16,
            fontweight="bold",
            color=color_text,
            y=0.98,
        )

        plt.show()
