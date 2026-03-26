"""
A class for extracting dominant shape patterns from clustered physiological signals for cardiac detection.

This module implements the DominantShapeExtractor class which identifies the most
common waveform pattern in a signal by clustering similar shapes and computing
the average shape from the main cluster. It supports both original and upsampled
signal resolutions for high-detail analysis.

Authors: Arnaud Poletto
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
from scipy.signal import find_peaks

from cinc.core.detection.utils.cardiac_detection.clustering.similarity_clustering import (
    SimilarityClusterer,
)


class DominantShapeExtractor:
    """A class for extracting dominant shape patterns from clustered physiological signals."""

    def __init__(
        self,
        config: Dict,
        debug_print: bool = False,
        debug_plot: bool = False,
    ) -> None:
        """
        Initialize the dominant shape extractor with configuration and debug settings.

        Args:
            config (Dict): The configuration dictionary with required parameters.
            debug_print (bool, optional): Whether to enable debug printing. Defaults to False.
            debug_plot (bool, optional): Whether to enable debug plotting. Defaults to False.

        Raises:
            ValueError: If 'shape_window_s' is missing from config.
            ValueError: If 'validate_shape_quality' is missing from config.
            ValueError: If 'max_n_zero_crossings' is missing from config.
            ValueError: If 'correlation_penalty_range' is missing from config.
            ValueError: If 'correlation_penalty_quantile_percent' is missing from config.
            ValueError: If 'max_lag_s' is missing from config.
            ValueError: If 'good_correlation_threshold' is missing from config.
        """
        if "shape_window_s" not in config:
            raise ValueError(
                "❌ 'shape_window_s' must be specified in the config dictionary."
            )
        if "validate_shape_quality" not in config:
            raise ValueError(
                "❌ 'validate_shape_quality' must be specified in the config dictionary."
            )
        if "zero_crossing_threshold_ratio" not in config:
            raise ValueError(
                "❌ 'zero_crossing_threshold_ratio' must be specified in the config dictionary."
            )
        if "max_n_zero_crossings" not in config:
            raise ValueError(
                "❌ 'max_n_zero_crossings' must be specified in the config dictionary."
            )
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

        super(DominantShapeExtractor, self).__init__()

        self.debug_print = debug_print
        self.debug_plot = debug_plot
        self.shape_window_s = config["shape_window_s"]
        self.validate_shape_quality = config["validate_shape_quality"]
        self.max_n_zero_crossings = config["max_n_zero_crossings"]
        self.zero_crossing_threshold_ratio = config["zero_crossing_threshold_ratio"]

        # Initialize similarity clusterer
        clusterer_config = {
            "correlation_penalty_range": config["correlation_penalty_range"],
            "correlation_penalty_quantile_percent": config[
                "correlation_penalty_quantile_percent"
            ],
            "max_lag_s": config["max_lag_s"],
            "good_correlation_threshold": config["good_correlation_threshold"],
        }
        self.similarity_clusterer = SimilarityClusterer(
            config=clusterer_config,
            debug_print=debug_print,
            debug_plot=debug_plot,
        )

    def _find_main_cluster(
        self,
        cluster_labels: np.ndarray,
    ) -> Optional[int]:
        """
        Find the cluster with the most events.

        Args:
            cluster_labels (np.ndarray): The array of cluster labels.

        Returns:
            cluster_id (Optional[int]): The ID of the main cluster, or None if no valid clusters found.
        """
        # Get valid clusters, excluding NaN and invalid labels
        valid_clusters = cluster_labels[~np.isnan(cluster_labels)]
        valid_clusters = valid_clusters[valid_clusters > 0]  # Exclude invalid labels
        if len(valid_clusters) == 0:
            if self.debug_print:
                print("⚠️  No valid clusters found.")
            return None

        # Find clusters with maximum count and take the first one in case of ties
        unique_clusters, counts = np.unique(valid_clusters, return_counts=True)
        max_count = np.max(counts)
        max_count_mask = counts == max_count
        tied_clusters = unique_clusters[max_count_mask]
        if len(tied_clusters) > 1:
            if self.debug_print:
                print(
                    f"⚠️  Multiple clusters tied for main cluster: {tied_clusters}. Selecting the first one."
                )
        main_cluster_id = tied_clusters[0]

        return int(main_cluster_id)

    def _compute_cluster_average_shape(
        self,
        signal_data: np.ndarray,
        upsampled_signal_data: np.ndarray,
        detections: np.ndarray,
        cluster_labels: np.ndarray,
        main_cluster_id: int,
        shape_window: Tuple[int, int],
        upsampled_factor: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compute the average shape for events in the main cluster.

        Args:
            signal_data (np.ndarray): The original signal data.
            upsampled_signal_data (np.ndarray): The upsampled signal data.
            detections (np.ndarray): The array of detection indices.
            cluster_labels (np.ndarray): The cluster label for each detection.
            main_cluster_id (int): The ID of the main cluster.
            shape_window (Tuple[int, int]): The window around each detection (start_offset, end_offset).
            upsampled_factor (int): The upsampling factor.

        Returns:
            average_shape (Optional[np.ndarray]): The average shape from the main cluster, or None if no valid shapes found.
            average_upsampled_shape (Optional[np.ndarray]): The average upsampled shape from the main cluster, or None if no valid shapes found.
        """
        main_cluster_indices = np.where(cluster_labels == main_cluster_id)[0]
        if len(main_cluster_indices) == 0:
            if self.debug_print:
                print("⚠️  No events found in the main cluster.")
            return None, None

        # Extract shapes
        shapes = []
        upsampled_shapes = []
        for i in main_cluster_indices:
            detection = detections[i]

            # No optimal shift here because we calibrate from detections
            window_start = detection + shape_window[0]
            window_end = detection + shape_window[1]
            upsampled_window_start = int(window_start * upsampled_factor)
            upsampled_window_end = int(window_end * upsampled_factor)
            if window_start < 0 or window_end >= len(signal_data):
                continue

            shape_segment = signal_data[window_start:window_end]
            upsampled_shape_segment = upsampled_signal_data[
                upsampled_window_start:upsampled_window_end
            ]
            if len(shape_segment) == 0 or len(upsampled_shape_segment) == 0:
                continue

            shapes.append(shape_segment)
            upsampled_shapes.append(upsampled_shape_segment)

        if len(shapes) == 0 or len(upsampled_shapes) == 0:
            if self.debug_print:
                print("⚠️  No valid shapes extracted from main cluster events.")
            return None, None

        # Compute average shape
        average_shape = np.mean(shapes, axis=0)
        average_upsampled_shape = np.mean(upsampled_shapes, axis=0)

        return average_shape, average_upsampled_shape
    
    def _validate_shape_quality(
        self,
        upsampled_dominant_shape: np.ndarray,
    ) -> bool:
        """
        Validate the quality of the dominant shape by detecting high-frequency artifacts.

        This method uses threshold-level crossings to identify high-frequency noise
        patterns that appear as rapid oscillations in the dominant shape. High-frequency
        sinusoidal noise will cross amplitude thresholds many times, while physiological
        cardiac signals have fewer, characteristic peaks and valleys.

        Args:
            upsampled_dominant_shape (np.ndarray): The dominant shape extracted from
                clustered cardiac events, at upsampled resolution.

        Returns:
            bool: True if the shape passes quality validation (likely physiological),
                False if rejected (likely high-frequency artifact).
        """
        # Detrend the shape (remove DC component)
        shape_detrended = upsampled_dominant_shape - np.mean(upsampled_dominant_shape)

        # Compute threshold levels as fraction of peak amplitudes
        # These thresholds help distinguish high-frequency oscillations from
        # the natural shape variations of cardiac waveforms
        ival_pos = self.zero_crossing_threshold_ratio * np.max(shape_detrended)
        ival_neg = self.zero_crossing_threshold_ratio * np.min(shape_detrended)

        # Count threshold crossings for each level
        # np.diff(np.sign(...)) detects where the signal crosses the threshold
        n_zero_crossings_pos = np.sum(np.diff(np.sign(shape_detrended - ival_pos)) != 0)
        n_zero_crossings_neg = np.sum(np.diff(np.sign(shape_detrended - ival_neg)) != 0)

        # Use minimum to be conservative - both sides must show excessive oscillation
        # This prevents rejection of asymmetric but valid cardiac shapes
        n_zero_crossings = min(n_zero_crossings_pos, n_zero_crossings_neg)

        # Reject if crossings exceed threshold
        if n_zero_crossings > self.max_n_zero_crossings:
            if self.debug_print:
                print(f"⚠️  Shape rejected: {n_zero_crossings} crossings "
                      f"(pos: {n_zero_crossings_pos}, neg: {n_zero_crossings_neg}, "
                      f"max allowed: {self.max_n_zero_crossings})")
            return False

        if self.debug_print:
            print(f"✅ Shape accepted: {n_zero_crossings} crossings "
                  f"(pos: {n_zero_crossings_pos}, neg: {n_zero_crossings_neg}, "
                  f"max allowed: {self.max_n_zero_crossings})")

        return True

    def run(
        self,
        signal_data: np.ndarray,
        upsampled_signal_data: np.ndarray,
        detections: np.ndarray,
        fs: float,
        upsampled_fs: float,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract the dominant shape pattern from clustered cardiac events.

        Args:
            signal_data (np.ndarray): The original signal data.
            upsampled_signal_data (np.ndarray): The upsampled signal data.
            detections (np.ndarray): The array of detection indices.
            fs (float): The sampling frequency in Hz.
            upsampled_fs (float): The upsampled sampling frequency in Hz.

        Returns:
            dominant_shape (Optional[np.ndarray]): The average shape from the main cluster, or None if no valid shape found.
            upsampled_dominant_shape (Optional[np.ndarray]): The average upsampled shape from the main cluster, or None if no valid shape found.
        """
        if len(detections) == 0:
            return None, None

        # Cluster events based on shape similarity
        shape_window = (
            int(self.shape_window_s[0] * fs),
            int(self.shape_window_s[1] * fs),
        )
        cluster_labels, shifts = self.similarity_clusterer.run(
            signal_data=signal_data,
            detections=detections,
            fs=fs,
            shape_window=shape_window,
        )

        # Find main cluster
        main_cluster_id = self._find_main_cluster(cluster_labels=cluster_labels)
        if main_cluster_id is None:
            if self.debug_print:
                print("⚠️  No valid main cluster found.")
            return None, None

        # Compute average shape for main cluster
        shape_window = (
            int(self.shape_window_s[0] * fs),
            int(self.shape_window_s[1] * fs),
        )
        upsampled_factor = int(upsampled_fs / fs)
        dominant_shape, upsampled_dominant_shape = self._compute_cluster_average_shape(
            signal_data=signal_data,
            upsampled_signal_data=upsampled_signal_data,
            detections=detections,
            cluster_labels=cluster_labels,
            main_cluster_id=main_cluster_id,
            shape_window=shape_window,
            upsampled_factor=upsampled_factor,
        )
        if dominant_shape is None or upsampled_dominant_shape is None:
            if self.debug_print:
                print(
                    "⚠️  No dominant shape could be extracted, returning empty result."
                )
            return None, None

        # Validate shape quality
        if self.validate_shape_quality:
            is_valid = self._validate_shape_quality(
                upsampled_dominant_shape=upsampled_dominant_shape
            )
            if not is_valid:
                if self.debug_print:
                    print("⚠️  Dominant shape failed quality validation.")
                return None, None

        # Plot dominant shapes if debug enabled
        if self.debug_plot:
            self._plot_dominant_shapes(dominant_shape, upsampled_dominant_shape)

        return dominant_shape, upsampled_dominant_shape

    def _plot_dominant_shapes(
        self,
        dominant_shape: np.ndarray,
        upsampled_dominant_shape: np.ndarray,
    ) -> None:
        """
        Visualize dominant shape patterns extracted from main cluster.

        Creates two subplots showing the dominant shape at original and upsampled
        resolutions, providing insight into the characteristic waveform pattern.

        Args:
            dominant_shape (np.ndarray): The average shape from main cluster at original resolution.
            upsampled_dominant_shape (np.ndarray): The average shape at higher resolution for detail.
        """
        # Define color palette
        color_dominant = "#d48cf0"
        color_upsampled = "#8e44ad"
        color_text = "#34495e"
        color_grid = "#bdc3c7"

        # Plot 1: Dominant shape at original resolution
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), dpi=100)
        fig.patch.set_facecolor("white")

        # Plot dominant shape at original resolution
        sample_indices = np.arange(len(dominant_shape))
        ax1.plot(
            sample_indices,
            dominant_shape,
            color=color_dominant,
            linewidth=2.5,
            label="Dominant Shape",
            zorder=2,
        )
        ax1.set_xlabel("Sample Index", fontsize=12, fontweight="bold", color=color_text)
        ax1.set_ylabel("Amplitude", fontsize=12, fontweight="bold", color=color_text)
        ax1.set_title(
            "Dominant Shape from Main Cluster",
            fontsize=15,
            fontweight="bold",
            color=color_text,
            pad=15,
        )

        # Position legend outside plot area at bottom right
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

        # Apply subtle grid for readability
        ax1.grid(
            True, alpha=0.2, linestyle="--", linewidth=1, color=color_grid, zorder=0
        )
        ax1.set_xlim(0, len(dominant_shape))

        # Minimize visual clutter by hiding top and right spines
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["left"].set_color(color_text)
        ax1.spines["bottom"].set_color(color_text)
        ax1.spines["left"].set_linewidth(1.5)
        ax1.spines["bottom"].set_linewidth(1.5)
        ax1.tick_params(colors=color_text, which="both", labelsize=10)

        # Plot 2: Upsampled dominant shape for higher detail
        upsampled_indices = np.arange(len(upsampled_dominant_shape))
        ax2.plot(
            upsampled_indices,
            upsampled_dominant_shape,
            color=color_upsampled,
            linewidth=2.5,
            label="Upsampled Dominant Shape",
            zorder=2,
        )
        ax2.set_xlabel("Sample Index", fontsize=12, fontweight="bold", color=color_text)
        ax2.set_ylabel("Amplitude", fontsize=12, fontweight="bold", color=color_text)
        ax2.set_title(
            "Upsampled Dominant Shape from Main Cluster",
            fontsize=15,
            fontweight="bold",
            color=color_text,
            pad=15,
        )

        # Position legend outside plot area at bottom right
        legend2 = ax2.legend(
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
        legend2.get_frame().set_linewidth(2)

        # Apply subtle grid for readability
        ax2.grid(
            True, alpha=0.2, linestyle="--", linewidth=1, color=color_grid, zorder=0
        )
        ax2.set_xlim(0, len(upsampled_dominant_shape))

        # Minimize visual clutter by hiding top and right spines
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_color(color_text)
        ax2.spines["bottom"].set_color(color_text)
        ax2.spines["left"].set_linewidth(1.5)
        ax2.spines["bottom"].set_linewidth(1.5)
        ax2.tick_params(colors=color_text, which="both", labelsize=10)

        # Adjust layout to accommodate external legends
        plt.tight_layout(rect=[0, 0.02, 1, 1])
        plt.show()
