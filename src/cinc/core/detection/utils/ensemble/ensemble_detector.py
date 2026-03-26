"""
Ensemble detector for combining detection results from multiple signals.

This module provides the EnsembleDetector class which performs peak detection
on ensemble likelihood signals and fills missing detections using gap analysis
and snapping to nearby maxima.

Authors: Arnaud Poletto
"""

import numpy as np
from typing import List, Dict
from scipy.signal import find_peaks

from cinc.core.detection.utils.interval.interval_detector import IntervalDetector
from cinc.core.detection.utils.fill.missing_detection_filler import MissingDetectionFiller


class EnsembleDetector:
    """A class for detecting events from ensemble likelihood signals."""

    def __init__(
        self,
        config: Dict,
        enable_detection_filling: bool,
        detect_both_phases: bool = False,
        cardiac_mode: bool = False,
        debug_print: bool = False,
    ) -> None:
        """
        Initialize the ensemble detector with configuration parameters.

        Args:
            config (Dict): The configuration dictionary with required parameters.
            detect_both_phases (bool, optional): Whether to detect both phases. Defaults to False.
            debug_print (bool, optional): Whether to enable debug printing. Defaults to False.

        Raises:
            ValueError: If 'distance_ratio' is missing from config.
            ValueError: If 'prominence' is missing from config.
            ValueError: If 'mad_threshold_lower' is missing from config.
            ValueError: If 'mad_threshold_upper' is missing from config.
            ValueError: If 'min_interval_s' is missing from config.
            ValueError: If 'max_interval_s' is missing from config.
            ValueError: If 'min_mad_s' is missing from config.
            ValueError: If 'max_mad_s' is missing from config.
            ValueError: If 'interval_estimation_window_s' is missing from config.
            ValueError: If 'gap_detection_multiplier' is missing from config.
            ValueError: If 'max_fill_count' is missing from config.
            ValueError: If 'min_snap_distance_s' is missing from config.
        """
        if "distance_ratio" not in config:
            raise ValueError(
                "❌ 'distance_ratio' must be specified in the config dictionary."
            )
        if "prominence" not in config:
            raise ValueError(
                "❌ 'prominence' must be specified in the config dictionary."
            )
        if "min_interval_s" not in config:
            raise ValueError(
                "❌ 'min_interval_s' must be specified in the config dictionary."
            )
        if "max_interval_s" not in config:
            raise ValueError(
                "❌ 'max_interval_s' must be specified in the config dictionary."
            )
        if "coarse_window_size" not in config:
            raise ValueError(
                "❌ 'coarse_window_size' must be specified in the config dictionary."
            )
        if "coarse_tolerance" not in config:
            raise ValueError(
                "❌ 'coarse_tolerance' must be specified in the config dictionary."
            )
        if "fine_tolerance" not in config:
            raise ValueError(
                "❌ 'fine_tolerance' must be specified in the config dictionary."
            )
        if "fine_window_sizes" not in config:
            raise ValueError(
                "❌ 'fine_window_sizes' must be specified in the config dictionary."
            )
        if "min_valid_ratio" not in config:
            raise ValueError(
                "❌ 'min_valid_ratio' must be specified in the config dictionary."
            )
        if "interval_estimation_window_s" not in config:
            raise ValueError(
                "❌ 'interval_estimation_window_s' must be specified in the config dictionary."
            )
        if "gap_detection_multiplier" not in config:
            raise ValueError(
                "❌ 'gap_detection_multiplier' must be specified in the config dictionary."
            )
        if "max_fill_count" not in config:
            raise ValueError(
                "❌ 'max_fill_count' must be specified in the config dictionary."
            )
        if "min_snap_distance_s" not in config:
            raise ValueError(
                "❌ 'min_snap_distance_s' must be specified in the config dictionary."
            )

        super(EnsembleDetector, self).__init__()

        self.detect_both_phases = detect_both_phases
        self.debug_print = debug_print
        self.distance_ratio = config["distance_ratio"]
        self.prominence = config["prominence"]
        self.enable_detection_filling = enable_detection_filling

        # Initialize interval detector
        interval_detector_config = {
            "min_interval_s": config["min_interval_s"],
            "max_interval_s": config["max_interval_s"],
            "coarse_window_size": config["coarse_window_size"],
            "coarse_tolerance": config["coarse_tolerance"],
            "fine_tolerance": config["fine_tolerance"],
            "fine_window_sizes": config["fine_window_sizes"],
            "min_valid_ratio": config["min_valid_ratio"],
        }
        self.interval_detector = IntervalDetector(
            config=interval_detector_config,
            cardiac_mode=cardiac_mode,
            debug_print=debug_print,
        )

        # Initialize missing detection filler
        missing_detection_filler_config = {
            "interval_estimation_window_s": config["interval_estimation_window_s"],
            "gap_detection_multiplier": config["gap_detection_multiplier"],
            "max_fill_count": config["max_fill_count"],
            "min_snap_distance_s": config["min_snap_distance_s"],
        }
        self.missing_detection_filler = MissingDetectionFiller(
            config=missing_detection_filler_config,
            debug_print=debug_print,
        )

    def _get_ensemble_results(
        self,
        likelihood: np.ndarray,
        global_median_interval_length_s: float,
        fs: float,
    ) -> Dict:
        """
        Compute ensemble detection results from a likelihood signal.

        This method detects peaks in the likelihood signal, fills missing
        detections using gap analysis with snapping to nearby maxima, filters
        out detections that create short intervals, and computes valid intervals.

        Args:
            likelihood (np.ndarray): The likelihood signal array.
            global_median_interval_length_s (float): The global median interval length in seconds.
            fs (float): The sampling frequency in Hz.

        Returns:
            results (Dict): The detection results containing detections, prominences,
            are_filled flags, and intervals.
        """
        # Detect confident peaks from detection probability
        distance = int(global_median_interval_length_s * fs * self.distance_ratio)
        detections, properties = find_peaks(
            likelihood,
            distance=distance,
            prominence=self.prominence,
        )
        prominences = properties["prominences"]

        # Detect all maxima without prominence filtering for snapping
        all_detections, all_properties = find_peaks(
            likelihood,
            distance=distance,
            prominence=0,
        )
        all_prominences = all_properties["prominences"]

        # Fill missing detections
        results = {
            "detections": detections,
            "prominences": prominences,
        }
        if self.enable_detection_filling:
            results = self.missing_detection_filler.run(
                results=results,
                fs=fs,
                snapping_detections=all_detections,
                snapping_prominences=all_prominences,
            )
        else:
            results["are_filled"] = np.zeros(len(detections), dtype=bool)

        # Compute intervals
        results = self.interval_detector.run(
            results=results,
            fs=fs,
        )

        return results

    def run(
        self,
        likelihood: np.ndarray | List[np.ndarray],
        global_median_interval_length_s: float,
        fs: float,
    ) -> Dict:
        """
        Run ensemble detection on likelihood signal(s).

        This method processes either a single likelihood signal or a pair of
        likelihood signals (for dual-phase detection) and returns detection
        results for each phase.

        Args:
            likelihood (Union[np.ndarray, List[np.ndarray]]): The likelihood signal
                or list of two likelihood signals for dual-phase detection.
            global_median_interval_length_s (float): The global median interval length in seconds.
            fs (float): The sampling frequency in Hz.

        Returns:
            ensemble_results (Dict): The ensemble detection results with phase_0 and phase_1 keys.
        """
        if self.detect_both_phases:
            phase_0_likelihood, phase_1_likelihood = likelihood
            phase_0_results = self._get_ensemble_results(
                likelihood=phase_0_likelihood,
                global_median_interval_length_s=global_median_interval_length_s,
                fs=fs,
            )
            phase_1_results = self._get_ensemble_results(
                likelihood=phase_1_likelihood,
                global_median_interval_length_s=global_median_interval_length_s,
                fs=fs,
            )

            return {
                "phase_0": phase_0_results,
                "phase_1": phase_1_results,
            }
        else:
            phase_0_results = self._get_ensemble_results(
                likelihood=likelihood,
                global_median_interval_length_s=global_median_interval_length_s,
                fs=fs,
            )

            return {
                "phase_0": phase_0_results,
                "phase_1": None,
            }
