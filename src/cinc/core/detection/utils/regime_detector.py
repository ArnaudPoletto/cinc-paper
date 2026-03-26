"""
A class for detecting processing and noise regimes in physiological signals.

This module implements the RegimeDetector class which identifies clean processing
regions and noisy regions in multi-channel signals using envelope analysis and
peak detection. It supports automatic regime splitting, merging, and overlap
handling for optimal signal processing.

Authors: Arnaud Poletto
"""

import numpy as np
from dataclasses import dataclass
from scipy.signal import find_peaks
from typing import List, Tuple, Dict
from matplotlib.patches import Patch
from matplotlib import pyplot as plt
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d


@dataclass
class Regime:
    """A data class representing a time regime with start and stop indices."""

    start: int
    stop: int

    @property
    def duration(self) -> int:
        """
        Calculate the duration of the regime in samples.

        Returns:
            duration (int): The duration in samples.
        """
        return self.stop - self.start

    def duration_s(self, fs: int) -> float:
        """
        Calculate the duration of the regime in seconds.

        Args:
            fs (int): Sampling frequency in Hz.

        Returns:
            duration (float): The duration in seconds.
        """
        return self.duration / fs


class RegimeDetector:
    """A class for detecting processing and noise regimes in physiological signals."""

    def __init__(
        self,
        config: Dict,
        debug_print: bool,
        debug_plot: bool,
    ) -> None:
        """
        Initialize the regime detector with configuration and debug settings.

        Args:
            config (Dict): Configuration dictionary with required parameters.
            debug_print (bool): Whether to enable debug printing.
            debug_plot (bool): Whether to enable debug plotting.

        Raises:
            ValueError: If required configuration parameters are missing.
        """
        # Validate parameters
        if "overlap_s" not in config:
            raise ValueError(
                "❌ 'overlap_s' must be specified in the config dictionary."
            )
        if "filter_window_s" not in config:
            raise ValueError(
                "❌ 'filter_window_s' must be specified in the config dictionary."
            )
        if "baseline_filter_window_s" not in config:
            raise ValueError(
                "❌ 'baseline_filter_window_s' must be specified in the config dictionary."
            )
        if "prominence_factor" not in config:
            raise ValueError(
                "❌ 'prominence_factor' must be specified in the config dictionary."
            )
        if "range_rel_height" not in config:
            raise ValueError(
                "❌ 'range_rel_height' must be specified in the config dictionary."
            )
        if "min_processing_regime_duration_s" not in config:
            raise ValueError(
                "❌ 'min_processing_regime_duration_s' must be specified in the config dictionary."
            )
        if "max_processing_regime_duration_s" not in config:
            raise ValueError(
                "❌ 'max_processing_regime_duration_s' must be specified in the config dictionary."
            )

        super(RegimeDetector, self).__init__()

        # Set debug modes
        self.debug_print = debug_print
        self.debug_plot = debug_plot

        # Set configuration parameters
        self.overlap_s = config["overlap_s"]
        self.filter_window_s = config["filter_window_s"]
        self.baseline_filter_window_s = config["baseline_filter_window_s"]
        self.prominence_factor = config["prominence_factor"]
        self.range_rel_height = config["range_rel_height"]
        self.min_processing_regime_duration_s = config[
            "min_processing_regime_duration_s"
        ]
        self.max_processing_regime_duration_s = config[
            "max_processing_regime_duration_s"
        ]

    def _compute_envelope_range(
        self,
        signal_data_list: List[np.ndarray],
        fs: float,
    ) -> np.ndarray:
        """
        Compute the envelope range across multiple signals.

        Args:
            signal_data_list (List[np.ndarray]): List of signal arrays.
            fs (float): Sampling frequency in Hz.

        Returns:
            normalized_range (np.ndarray): The normalized envelope range.
        """
        # Compute upper and lower envelopes for each signal
        filter_window = int(self.filter_window_s * fs)
        uppers = []
        lowers = []
        for signal_data in signal_data_list:
            upper = maximum_filter1d(signal_data, size=filter_window)
            lower = minimum_filter1d(signal_data, size=filter_window)

            # Smooth the envelopes
            upper = uniform_filter1d(upper, size=filter_window)
            lower = uniform_filter1d(lower, size=filter_window)
            uppers.append(upper)
            lowers.append(lower)

        # Range is the difference between max upper and min lower across signals
        raw_range = np.max(uppers, axis=0) - np.min(lowers, axis=0)

        # Remove baseline
        baseline_window = int(self.baseline_filter_window_s * fs)
        baseline = minimum_filter1d(raw_range, size=baseline_window)
        baseline = uniform_filter1d(baseline, size=baseline_window)

        normalized_range = raw_range - baseline

        return normalized_range

    def _merge_adjacent_regimes(
        self,
        regimes: List[Regime],
    ) -> List[Regime]:
        """
        Merge regimes that are adjacent (touching or overlapping).

        Args:
            regimes (List[Regime]): List of regime objects to merge.

        Returns:
            merged_regimes (List[Regime]): List of merged regimes.
        """
        if len(regimes) == 0:
            return []

        sorted_regimes = sorted(regimes, key=lambda r: r.start)
        merged_regimes = [sorted_regimes[0]]

        for current in sorted_regimes[1:]:
            # Check if current regime is adjacent to or overlaps with the last merged regime
            if current.start <= merged_regimes[-1].stop:
                # Merge by extending the stop to the max of both
                merged_regimes[-1] = Regime(
                    start=merged_regimes[-1].start,
                    stop=max(merged_regimes[-1].stop, current.stop),
                )
            else:
                merged_regimes.append(current)

        return merged_regimes

    def _compute_noise_prominence(self, signal_range: np.ndarray) -> float:
        """
        Compute prominence threshold for noise peak detection.

        Args:
            signal_range (np.ndarray): The signal envelope range.

        Returns:
            prominence (float): The prominence threshold value.
        """
        median = np.median(signal_range)
        mad = np.median(np.abs(signal_range - median))
        robust_std = mad * 1.4826

        return robust_std * self.prominence_factor

    def _detect_initial_noise_regimes(self, signal_range: np.ndarray) -> List[Regime]:
        """
        Detect initial noise regions from the signal range.

        Args:
            signal_range (np.ndarray): The signal envelope range.

        Returns:
            noise_regimes (List[Regime]): List of detected noise regimes.
        """
        prominence = self._compute_noise_prominence(signal_range)
        peaks, properties = find_peaks(
            signal_range,
            prominence=prominence,
            width=0,
            rel_height=self.range_rel_height,
        )

        if len(peaks) == 0:
            return []

        noise_regimes = []
        for left_ips, right_ips in zip(properties["left_ips"], properties["right_ips"]):
            start = int(left_ips)
            stop = int(right_ips)
            noise_regimes.append(Regime(start=start, stop=stop))

        # Merge adjacent noise regions
        noise_regimes = self._merge_adjacent_regimes(noise_regimes)

        return noise_regimes

    def _invert_regimes(
        self,
        noise_regimes: List[Regime],
        signal_length: int,
    ) -> List[Regime]:
        """
        Invert noise regimes to get processing (clean) regimes.

        Args:
            noise_regimes (List[Regime]): List of noise regimes.
            signal_length (int): Total length of the signal in samples.

        Returns:
            processing_regimes (List[Regime]): List of processing regimes (gaps between noise).
        """
        if len(noise_regimes) == 0:
            return [Regime(start=0, stop=signal_length)]

        processing_regimes = []

        # Sort noise regimes by start
        sorted_noise = sorted(noise_regimes, key=lambda r: r.start)

        # Add region before first noise (if exists)
        if sorted_noise[0].start > 0:
            processing_regimes.append(Regime(start=0, stop=sorted_noise[0].start))

        # Add regions between noise regimes
        for i in range(len(sorted_noise) - 1):
            gap_start = sorted_noise[i].stop
            gap_stop = sorted_noise[i + 1].start
            if gap_start < gap_stop:
                processing_regimes.append(Regime(start=gap_start, stop=gap_stop))

        # Add region after last noise (if exists)
        if sorted_noise[-1].stop < signal_length:
            processing_regimes.append(
                Regime(start=sorted_noise[-1].stop, stop=signal_length)
            )

        return processing_regimes

    def _filter_short_processing_regimes(
        self,
        processing_regimes: List[Regime],
        min_duration: int,
    ) -> Tuple[List[Regime], List[Regime]]:
        """
        Filter out processing regimes that are too short.

        Args:
            processing_regimes (List[Regime]): List of processing regimes.
            min_duration (int): Minimum duration in samples.

        Returns:
            valid_processing (List[Regime]): List of valid processing regimes.
            short_as_noise (List[Regime]): List of short regimes converted to noise.
        """
        valid_processing = []
        short_as_noise = []

        for regime in processing_regimes:
            if regime.duration >= min_duration:
                valid_processing.append(regime)
            else:
                # Convert short processing regime to noise
                short_as_noise.append(regime)

        return valid_processing, short_as_noise

    def _split_large_regime(
        self,
        regime: Regime,
        max_duration: int,
    ) -> List[Regime]:
        """
        Split a regime that exceeds max_duration into smaller regimes.

        Args:
            regime (Regime): The regime to split.
            max_duration (int): Maximum duration in samples.

        Returns:
            split_regimes (List[Regime]): List of split regimes.
        """
        if regime.duration <= max_duration:
            return [regime]

        # Calculate number of splits needed
        n_splits = int(np.ceil(regime.duration / max_duration))
        split_size = regime.duration // n_splits

        split_regimes = []
        pos = regime.start
        for i in range(n_splits):
            split_stop = pos + split_size if i < n_splits - 1 else regime.stop
            split_regimes.append(Regime(start=pos, stop=split_stop))
            pos = split_stop

        return split_regimes

    def _split_large_processing_regimes(
        self,
        processing_regimes: List[Regime],
        max_duration: int,
    ) -> List[Regime]:
        """
        Split processing regimes that exceed max_duration.

        Args:
            processing_regimes (List[Regime]): List of processing regimes.
            max_duration (int): Maximum duration in samples.

        Returns:
            split_regimes (List[Regime]): List of split processing regimes.
        """
        split_regimes = []
        for regime in processing_regimes:
            split_regimes.extend(self._split_large_regime(regime, max_duration))

        return split_regimes

    def _add_overlap_to_regimes(
        self,
        regimes: List[Regime],
        signal_length: int,
        overlap_samples: int,
    ) -> List[Regime]:
        """
        Add overlap between adjacent processing regimes.

        Args:
            regimes (List[Regime]): List of processing regimes.
            signal_length (int): Total signal length in samples.
            overlap_samples (int): Overlap duration in samples.

        Returns:
            overlapped_regimes (List[Regime]): List of regimes with added overlaps.
        """
        if len(regimes) <= 1:
            return regimes

        # Sort regimes by start index
        regimes = sorted(regimes, key=lambda r: r.start)

        half_overlap = overlap_samples // 2
        overlapped_regimes = []
        for i, regime in enumerate(regimes):
            new_start = regime.start
            new_stop = regime.stop

            # Check if this regime is adjacent to the previous one
            if i > 0 and regimes[i - 1].stop == regime.start:
                # Extend start backward by half overlap
                new_start = max(0, regime.start - half_overlap)

            # Check if this regime is adjacent to the next one
            if i < len(regimes) - 1 and regime.stop == regimes[i + 1].start:
                # Extend stop forward by half overlap
                new_stop = min(signal_length, regime.stop + half_overlap)

            overlapped_regimes.append(Regime(start=new_start, stop=new_stop))

        return overlapped_regimes

    def run(
        self,
        signal_data_list: List[np.ndarray],
        fs: int,
    ) -> Tuple[List[Regime], List[Regime]]:
        """
        Detect processing and noise regimes from signal data.

        Args:
            signal_data_list (List[np.ndarray]): List of signal arrays to analyze.
            fs (int): Sampling frequency in Hz.

        Returns:
            processing_regimes (List[Regime]): List of detected processing regimes.
            noise_regimes (List[Regime]): List of detected noise regimes.
        """
        signal_length = signal_data_list[0].shape[0]
        min_samples = int(self.min_processing_regime_duration_s * fs)
        max_samples = int(self.max_processing_regime_duration_s * fs)
        overlap_samples = int(self.overlap_s * fs)

        # Compute envelope range
        signal_range = self._compute_envelope_range(
            signal_data_list=signal_data_list, fs=fs
        )

        # Detect initial noise regions from peaks in the envelope range
        initial_noise_regimes = self._detect_initial_noise_regimes(
            signal_range=signal_range,
        )

        # Invert noise regions to get processing (clean) regions
        processing_regimes = self._invert_regimes(
            noise_regimes=initial_noise_regimes,
            signal_length=signal_length,
        )

        # Filter out processing regions that are too short
        # These become additional noise regions
        processing_regimes, short_as_noise = self._filter_short_processing_regimes(
            processing_regimes=processing_regimes,
            min_duration=min_samples,
        )

        # Merge the new noise regions with initial noise
        all_noise_regimes = initial_noise_regimes + short_as_noise
        noise_regimes = self._merge_adjacent_regimes(all_noise_regimes)

        # Split large processing regimes if they exceed max duration
        processing_regimes = self._split_large_processing_regimes(
            processing_regimes=processing_regimes,
            max_duration=max_samples,
        )

        # Add overlap between adjacent processing regimes
        processing_regimes = self._add_overlap_to_regimes(
            regimes=processing_regimes,
            signal_length=signal_length,
            overlap_samples=overlap_samples,
        )

        # Plot results if debug enabled
        if self.debug_plot:
            self._plot_results(
                signal_data_list=signal_data_list,
                signal_range=signal_range,
                noise_regimes=noise_regimes,
                processing_regimes=processing_regimes,
                fs=fs,
            )

        return processing_regimes, noise_regimes

    def _plot_results(
        self,
        signal_data_list: List[np.ndarray],
        signal_range: np.ndarray,
        noise_regimes: List[Regime],
        processing_regimes: List[Regime],
        fs: int,
    ) -> None:
        """
        Plot the signal, envelope range, and detected regimes.

        Args:
            signal_data_list (List[np.ndarray]): List of signal arrays.
            signal_range (np.ndarray): The signal envelope range.
            noise_regimes (List[Regime]): List of detected noise regimes.
            processing_regimes (List[Regime]): List of detected processing regimes.
            fs (int): Sampling frequency in Hz.
        """
        fig, axes = plt.subplots(
            2, 1, figsize=(14, 6), sharex=True, height_ratios=[2, 1]
        )

        # Plot 1: Signal and Envelope Range
        ax1 = axes[0]
        for i, signal_data in enumerate(signal_data_list):
            ax1.plot(signal_data, alpha=0.5, label="Signal" if i == 0 else None)
        ax1.plot(signal_range, color="red", alpha=0.8, label="Envelope Range")
        ax1.set_ylabel("Amplitude")
        ax1.legend(loc="upper right", bbox_to_anchor=(1.18, 1))
        ax1.set_title("Signal and Envelope Range")

        # Plot 2: Combined regime view
        ax2 = axes[1]

        # Layer 1: Processing regimes
        processing_colors = ["#4CAF50", "#81C784"]
        for i, regime in enumerate(processing_regimes):
            y_pos = 0.0 if i % 2 == 0 else 0.33

            ax2.broken_barh(
                [(regime.start, regime.stop - regime.start)],
                (y_pos, 0.3),
                facecolors=processing_colors[i % 2],
                edgecolors="black",
                linewidth=0.5,
                alpha=0.8,
            )
            mid = (regime.start + regime.stop) / 2
            duration_s = (regime.stop - regime.start) / fs
            ax2.text(
                mid,
                y_pos + 0.15,
                f"{duration_s:.0f}s",
                ha="center",
                va="center",
                fontsize=8,
            )

        # Highlight overlap zones
        for i in range(len(processing_regimes) - 1):
            curr = processing_regimes[i]
            next_reg = processing_regimes[i + 1]

            # Check if they overlap
            overlap_start = max(curr.start, next_reg.start)
            overlap_stop = min(curr.stop, next_reg.stop)
            if overlap_start < overlap_stop:
                ax2.broken_barh(
                    [(overlap_start, overlap_stop - overlap_start)],
                    (0.0, 0.63),
                    facecolors="none",
                    edgecolors="orange",
                    linewidth=2,
                    linestyle="--",
                )

        # Layer 2: Noise regimes
        for i, regime in enumerate(noise_regimes):
            is_absorbed = any(
                p.start <= regime.start and regime.stop <= p.stop
                for p in processing_regimes
            )

            ax2.broken_barh(
                [(regime.start, regime.stop - regime.start)],
                (0.7, 0.3),
                facecolors="#E57373",
                edgecolors="black",
                linewidth=0.5,
                hatch="///" if is_absorbed else None,
                alpha=0.9,
            )
            mid = (regime.start + regime.stop) / 2
            duration_s = (regime.stop - regime.start) / fs
            ax2.text(
                mid, 0.85, f"{duration_s:.0f}s", ha="center", va="center", fontsize=8
            )

        # Legend
        legend_elements = [
            Patch(facecolor="#4CAF50", edgecolor="black", label="Processing (even)"),
            Patch(facecolor="#81C784", edgecolor="black", label="Processing (odd)"),
            Patch(
                facecolor="none",
                edgecolor="orange",
                linewidth=2,
                linestyle="--",
                label="Overlap zone",
            ),
            Patch(facecolor="#E57373", edgecolor="black", label="Noise (isolated)"),
            Patch(
                facecolor="#E57373",
                edgecolor="black",
                hatch="///",
                label="Noise (absorbed)",
            ),
        ]
        ax2.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.22, 1))

        ax2.set_ylabel("Regimes")
        ax2.set_yticks([0.15, 0.48, 0.85])
        ax2.set_yticklabels(["Proc (even)", "Proc (odd)", "Noise"])
        ax2.set_ylim(0, 1)
        xticks = ax2.get_xticks()
        print(xticks)
        ax2.set_xticks(xticks)
        ax2.set_xticklabels([f"{x/fs:.0f}" for x in xticks])
        ax2.set_xlabel("Seconds")

        plt.tight_layout()
        plt.show()
