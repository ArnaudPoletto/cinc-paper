"""
Template matcher for cardiac detection using similarity-based peak matching.

This module provides the TemplateMatcher class which detects candidate peaks in
upsampled signals and matches them against a dominant shape template using
similarity analysis.

Authors: Arnaud Poletto
"""

import numpy as np
from typing import Dict
from scipy.signal import find_peaks
from .similarity_analyzer import SimilarityAnalyzer


class TemplateMatcher:
    """
    A class for detecting and matching cardiac peaks using template matching with similarity analysis.
    """

    def __init__(
        self,
        config: Dict,
        debug_print: bool = False,
        debug_plot: bool = False,
    ) -> None:
        """
        Initialize the template matcher with configuration parameters.

        Args:
            config (Dict): The configuration dictionary with required parameters.
            debug_print (bool, optional): Whether to enable debug printing. Defaults to False.
            debug_plot (bool, optional): Whether to enable debug plotting. Defaults to False.

        Raises:
            ValueError: If 'min_separation_s' is missing from config.
            ValueError: If 'prominence_std_factor' is missing from config.
            ValueError: If 'prominence_wlen_s' is missing from config.
            ValueError: If 'shape_window_s' is missing from config.
            ValueError: If 'max_lag_s' is missing from config.
            ValueError: If 'correlation_penalty_range' is missing from config.
            ValueError: If 'correlation_penalty_quantile_percent' is missing from config.
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
        if "shape_window_s" not in config:
            raise ValueError(
                "❌ 'shape_window_s' must be specified in the config dictionary."
            )
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

        super(TemplateMatcher, self).__init__()

        self.debug_print = debug_print
        self.debug_plot = debug_plot
        self.min_separation_s = config["min_separation_s"]
        self.prominence_std_factor = config["prominence_std_factor"]
        self.prominence_wlen_s = config["prominence_wlen_s"]
        self.shape_window_s = config["shape_window_s"]

        similarity_analyzer_config = {
            "max_lag_s": config["max_lag_s"],
            "correlation_penalty_range": config["correlation_penalty_range"],
            "correlation_penalty_quantile_percent": config[
                "correlation_penalty_quantile_percent"
            ],
        }
        self.similarity_analyzer = SimilarityAnalyzer(
            config=similarity_analyzer_config,
            debug_print=debug_print,
            debug_plot=debug_plot,
        )

    def run(
        self,
        upsampled_signal_data: np.ndarray,
        upsampled_dominant_shape: np.ndarray,
        upsampled_fs: float,
        sign: int,
    ) -> Dict:
        """
        Run template matching on the upsampled signal to detect cardiac peaks.

        This method detects candidate peaks in the upsampled signal using peak
        detection with prominence filtering, then matches them against the
        dominant shape template using similarity analysis.

        Args:
            upsampled_signal_data (np.ndarray): The upsampled signal data array.
            upsampled_dominant_shape (np.ndarray): The dominant shape template array.
            upsampled_fs (float): The upsampled sampling frequency in Hz.
            sign (int): The sign of peaks to detect (1 for positive, -1 for negative).

        Returns:
            similarity_results (Dict): A dictionary containing detected peaks and their prominences.
        """
        # Detect all candidate peaks in the upsampled signal
        distance = int(self.min_separation_s * upsampled_fs)
        min_prominence = np.std(upsampled_signal_data) * self.prominence_std_factor
        wlen = int(self.prominence_wlen_s * upsampled_fs)
        signed_upsampled_signal_data = (
            upsampled_signal_data if sign > 0 else -upsampled_signal_data
        )
        upsampled_detections, properties = find_peaks(
            signed_upsampled_signal_data,
            distance=distance,
            prominence=min_prominence,
            wlen=wlen,
        )
        upsampled_prominences = properties["prominences"]
        if upsampled_detections.shape[0] == 0:
            return {
                "detections": np.array([], dtype=int),
                "prominences": np.array([], dtype=float),
                "shifts": np.array([], dtype=int),
                "correlations": np.array([], dtype=float),
            }

        # Run similarity analysis to match detected peaks to the dominant shape
        upsampled_shape_window = (
            int(self.shape_window_s[0] * upsampled_fs),
            int(self.shape_window_s[1] * upsampled_fs),
        )
        shape_offset = upsampled_shape_window[0]
        similarity_results = self.similarity_analyzer.run(
            upsampled_signal_data=upsampled_signal_data,
            upsampled_detections=upsampled_detections,
            upsampled_prominences=upsampled_prominences,
            upsampled_fs=upsampled_fs,
            upsampled_reference_shape=upsampled_dominant_shape,
            shape_offset=shape_offset,
        )

        return similarity_results
