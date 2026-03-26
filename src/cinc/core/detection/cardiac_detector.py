"""
A class for detecting cardiac events from physiological signals.

This module implements the CardiacDetector class which uses template matching
to detect cardiac cycles from ECG or PPG signals. It supports ensemble methods
to combine detections from multiple signals and includes signal preprocessing,
template extraction, correlation-based detection, and interval detection.

Authors: Arnaud Poletto
"""

import numpy as np
from tqdm import tqdm
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from cinc.core.detection.utils.plot import plot_detections
from cinc.utils.config import dict_to_namespace, load_config
from cinc.core.detection.utils.regime_detector import Regime, RegimeDetector
from cinc.core.detection.utils.ensemble.ensemble_detector import EnsembleDetector
from cinc.core.detection.utils.interval.interval_detector import IntervalDetector
from cinc.core.detection.utils.fill.missing_detection_filler import (
    MissingDetectionFiller,
)
from cinc.core.detection.utils.cardiac_detection.signal_score import (
    get_processed_signal_score_dict,
)
from cinc.core.detection.utils.ensemble.ensemble_likelihood_estimator import (
    EnsembleLikelihoodEstimator,
)
from cinc.core.detection.utils.cardiac_detection.clustering.dominant_shape import (
    DominantShapeExtractor,
)
from cinc.core.detection.utils.cardiac_detection.template_matching.template_matcher import (
    TemplateMatcher,
)
from cinc.core.detection.utils.cardiac_detection.detections.detections_estimator import (
    DetectionsEstimator,
)
from cinc.core.detection.utils.sort_results import (
    sort_detection_results,
    sort_phase_results,
)


class CardiacDetector:
    """A class for detecting cardiac events from physiological signals using template matching."""

    def __init__(
        self,
        config_file_path: str,
        debug_plot: bool = False,
        debug_print: bool = False,
    ) -> None:
        """
        Initialize the cardiac detector with configuration and debug settings.

        Args:
            config_file_path (str): Path to the YAML configuration file.
            debug_plot (bool): Whether to enable debug plotting. Defaults to False.
            debug_print (bool): Whether to enable debug printing. Defaults to False.
        """
        super(CardiacDetector, self).__init__()

        self.debug_plot = debug_plot
        self.debug_print = debug_print

        # Load configuration and convert to namespace for attribute access
        config = load_config(config_file_path)
        for parameter, value in config.items():
            setattr(self, parameter, dict_to_namespace(value))
        self.config = config

        # Initialize detections estimator
        detections_estimator_config = {
            "min_separation_s": self.detections_estimation.min_separation_s,
            "prominence_std_factor": self.detections_estimation.prominence_std_factor,
            "prominence_wlen_s": self.detections_estimation.prominence_wlen_s,
        }
        self.detections_estimator = DetectionsEstimator(
            config=detections_estimator_config,
            debug_plot=debug_plot,
        )

        # Initialize dominant shape extractor
        dominant_shape_extractor_config = {
            "shape_window_s": self.dominant_shape_extracting.shape_window_s,
            "correlation_penalty_range": self.dominant_shape_extracting.correlation_penalty_range,
            "correlation_penalty_quantile_percent": self.dominant_shape_extracting.correlation_penalty_quantile_percent,
            "max_lag_s": self.dominant_shape_extracting.max_lag_s,
            "good_correlation_threshold": self.dominant_shape_extracting.good_correlation_threshold,
            "validate_shape_quality": self.dominant_shape_extracting.validate_shape_quality,
            "max_n_zero_crossings": self.dominant_shape_extracting.max_n_zero_crossings,
            "zero_crossing_threshold_ratio": self.dominant_shape_extracting.zero_crossing_threshold_ratio,
        }
        self.dominant_shape_extractor = DominantShapeExtractor(
            config=dominant_shape_extractor_config,
            debug_plot=debug_plot,
            debug_print=debug_print,
        )

        # Initialize template matcher
        template_matcher_config = {
            "min_separation_s": self.template_matching.min_separation_s,
            "prominence_std_factor": self.template_matching.prominence_std_factor,
            "prominence_wlen_s": self.template_matching.prominence_wlen_s,
            "shape_window_s": self.dominant_shape_extracting.shape_window_s,
            "max_lag_s": self.dominant_shape_extracting.max_lag_s,
            "correlation_penalty_range": self.dominant_shape_extracting.correlation_penalty_range,
            "correlation_penalty_quantile_percent": self.dominant_shape_extracting.correlation_penalty_quantile_percent,
        }
        self.template_matcher = TemplateMatcher(
            config=template_matcher_config,
            debug_print=debug_print,
            debug_plot=debug_plot,
        )

        # Initialize missing detection filler
        missing_detection_filler_config = {
            "interval_estimation_window_s": self.detection_filling.interval_estimation_window_s,
            "gap_detection_multiplier": self.detection_filling.gap_detection_multiplier,
            "max_fill_count": self.detection_filling.max_fill_count,
            "min_snap_distance_s": self.detection_filling.min_snap_distance_s,
        }
        self.missing_detection_filler = MissingDetectionFiller(
            config=missing_detection_filler_config,
            debug_print=debug_print,
        )

        # Initialize interval detector
        interval_detector_config = {
            "min_interval_s": self.interval_detection.min_interval_s,
            "max_interval_s": self.interval_detection.max_interval_s,
            "coarse_window_size": self.interval_detection.coarse_window_size,
            "coarse_tolerance": self.interval_detection.coarse_tolerance,
            "fine_tolerance": self.interval_detection.fine_tolerance,
            "fine_window_sizes": self.interval_detection.fine_window_sizes,
            "min_valid_ratio": self.interval_detection.min_valid_ratio,
        }
        self.interval_detector = IntervalDetector(
            config=interval_detector_config,
            cardiac_mode=True,
            debug_print=debug_print,
        )

        # Initialize ensemble likelihood estimator
        ensemble_likelihood_estimator_config = {
            "min_tolerance_s": self.likelihood_estimation.min_tolerance_s,
            "max_tolerance_interval_ratio": self.likelihood_estimation.max_tolerance_interval_ratio,
            "mad_threshold": self.likelihood_estimation.mad_threshold,
            "weight_temperature": self.likelihood_estimation.weight_temperature,
            "default_sigma_s": self.likelihood_estimation.default_sigma_s,
            "sigma_scaling_factor": self.likelihood_estimation.sigma_scaling_factor,
            "min_sigma_s": self.likelihood_estimation.min_sigma_s,
            "max_sigma_s": self.likelihood_estimation.max_sigma_s,
            "processed_signal_alpha": self.likelihood_estimation.processed_signal_alpha,
            "prominence_alpha": self.likelihood_estimation.prominence_alpha,
            "synchronize_detections": self.likelihood_estimation.synchronize_detections,
        }
        self.ensemble_likelihood_estimator = EnsembleLikelihoodEstimator(
            config=ensemble_likelihood_estimator_config,
            detect_both_phases=False,
            debug_print=debug_print,
        )

        # Initialize ensemble detector
        ensemble_detector_config = {
            "distance_ratio": self.ensemble_detection.distance_ratio,
            "prominence": self.ensemble_detection.prominence,
            "interval_estimation_window_s": self.detection_filling.interval_estimation_window_s,
            "gap_detection_multiplier": self.detection_filling.gap_detection_multiplier,
            "max_fill_count": self.detection_filling.max_fill_count,
            "min_snap_distance_s": self.detection_filling.min_snap_distance_s,
        }
        ensemble_detector_config.update(interval_detector_config)
        self.ensemble_detector = EnsembleDetector(
            config=ensemble_detector_config,
            enable_detection_filling=self.detection_filling.enable,
            detect_both_phases=False,
            cardiac_mode=True,
            debug_print=debug_print,
        )

        # Initialize regime detector
        regime_detection_config = {
            "overlap_s": self.regime_detection.overlap_s,
            "filter_window_s": self.regime_detection.filter_window_s,
            "baseline_filter_window_s": self.regime_detection.baseline_filter_window_s,
            "prominence_factor": self.regime_detection.prominence_factor,
            "range_rel_height": self.regime_detection.range_rel_height,
            "min_processing_regime_duration_s": self.regime_detection.min_processing_regime_duration_s,
            "max_processing_regime_duration_s": self.regime_detection.max_processing_regime_duration_s,
        }
        self.regime_detector = RegimeDetector(
            config=regime_detection_config,
            debug_print=debug_print,
            debug_plot=debug_plot,
        )

    def _process_ensemble(
        self,
        signal_results_list: List[Dict],
        upsampled_signal_data_list: List[np.ndarray],
        upsampled_fs: float,
        force_synchronization: bool = False,
    ) -> Dict:
        """
        Combine individual signal results using ensemble detection.

        Args:
            signal_results_list (List[Dict]): List of detection results from individual signals.
            upsampled_signal_data_list (List[np.ndarray]): List of upsampled signal data arrays.
            upsampled_fs (float): The upsampled sampling frequency in Hz.

        Returns:
            ensemble_results (Dict): Ensemble detection results for phase_0.
        """
        # Compute likelihood and global parameters (no signs for cardiac)
        signal_length = upsampled_signal_data_list[0].shape[0]
        likelihood, global_median_interval_length_s, _ = (
            self.ensemble_likelihood_estimator.run(
                results_list=signal_results_list,
                signal_length=signal_length,
                fs=upsampled_fs,
                force_synchronization=force_synchronization,
            )
        )

        if likelihood is None or global_median_interval_length_s is None:
            if self.debug_print:
                print(
                    "⚠️  Ensemble likelihood estimation failed, returning empty result."
                )
            return {
                "phase_0": {
                    "detections": np.array([], dtype=int),
                    "prominences": np.array([], dtype=float),
                    "are_filled": np.array([], dtype=bool),
                    "intervals": np.array([], dtype=int).reshape(0, 2),
                },
                "phase_1": None,
            }

        # Run ensemble detection
        ensemble_results = self.ensemble_detector.run(
            likelihood=likelihood,
            global_median_interval_length_s=global_median_interval_length_s,
            fs=upsampled_fs,
        )

        # If not enough intervals detected, run again with forced synchronization
        n_estimated_intervals = (
            1.0 / global_median_interval_length_s * (signal_length / upsampled_fs)
        ) - 1
        n_intervals = ensemble_results["phase_0"]["intervals"].shape[0]
        if not force_synchronization and (
            n_intervals
            < (self.ensemble_detection.min_ratio_forced_sync * n_estimated_intervals)
        ):
            if self.debug_print:
                print(
                    "⚠️  Not enough intervals detected in ensemble, re-running with forced synchronization."
                )

            return self._process_ensemble(
                signal_results_list=signal_results_list,
                upsampled_signal_data_list=upsampled_signal_data_list,
                upsampled_fs=upsampled_fs,
                force_synchronization=True,
            )

        # Plot detections if debug enabled
        if self.debug_plot:
            plot_detections(
                signal_data=likelihood,
                results=ensemble_results["phase_0"],
                fs=upsampled_fs,
            )

        # Debug printing
        if self.debug_print:
            phase_0_results = ensemble_results["phase_0"]
            print("✅ Phase 0 results with keys:")
            for key in phase_0_results:
                shape_str = (
                    f", shape: {phase_0_results[key].shape}"
                    if isinstance(phase_0_results[key], np.ndarray)
                    else ""
                )
                print(f"\t- {key} (phase 0): {type(phase_0_results[key])}{shape_str}")

        return ensemble_results

    def _merge_chunked_results_list(
        self,
        chunked_results_list: List[Dict],
        start_indices: np.ndarray,
        stop_indices: np.ndarray,
        fs: float,
    ) -> Dict:
        """
        Merge detection results from multiple chunks into a single unified result.

        Note: For cardiac detection, only phase_0 is processed (no phase_1).

        Args:
            chunked_results_list (List[Dict]): List of detection results from individual chunks.
            start_indices (np.ndarray): Array of start indices for each chunk.
            fs (float): The sampling frequency in Hz.

        Returns:
            merged_chunked_results (Dict): Merged detection results with adjusted indices.
        """
        merged_chunked_results = {
            "phase_0": {
                "detections": np.array([], dtype=int),
                "prominences": np.array([], dtype=float),
                "are_filled": np.array([], dtype=bool),
            },
            "phase_1": None,
        }

        for chunked_results, start_idx in zip(chunked_results_list, start_indices):
            # Only process phase_0 for cardiac detection
            phase_results = chunked_results["phase_0"]

            # Validate phase results existence
            if phase_results is None:
                if self.debug_print:
                    print(
                        f"⚠️  Chunk at index {start_idx} has no results for phase_0, skipping."
                    )
                continue

            # Validate phase results required keys
            required_keys = ["detections", "prominences", "are_filled"]
            if not all(key in phase_results for key in required_keys):
                if self.debug_print:
                    print(
                        f"⚠️  Chunk at index {start_idx} missing required keys for phase_0, skipping."
                    )
                continue

            # Merge phase results
            merged_chunked_results["phase_0"] = {
                "detections": np.concatenate(
                    [
                        merged_chunked_results["phase_0"]["detections"],
                        phase_results["detections"]
                        + start_idx,  # Offset by chunk start index
                    ]
                ),
                "prominences": np.concatenate(
                    [
                        merged_chunked_results["phase_0"]["prominences"],
                        phase_results["prominences"],
                    ]
                ),
                "are_filled": np.concatenate(
                    [
                        merged_chunked_results["phase_0"]["are_filled"],
                        phase_results["are_filled"],
                    ]
                ),
            }

        # Recompute intervals
        merged_chunked_results["phase_0"] = sort_detection_results(
            merged_chunked_results["phase_0"]
        )
        merged_chunked_results["phase_0"] = self.interval_detector.run(
            results=merged_chunked_results["phase_0"],
            fs=fs,
        )

        # Append processed signal score if available
        chunks_data = {
            "starts": start_indices,
            "stops": stop_indices,
            "processed_signal_scores": [
                chunked_results["processed_signal_score"]
                if "processed_signal_score" in chunked_results
                else None
                for chunked_results in chunked_results_list
            ],
        }

        # Remove processed signal scores if all are None
        if all(score is None for score in chunks_data["processed_signal_scores"]):
            chunks_data.pop("processed_signal_scores")

        merged_chunked_results["chunks"] = chunks_data

        return merged_chunked_results

    def run(
        self,
        signal_data: np.ndarray,
        upsampled_signal_data: Optional[np.ndarray],
        fs: float,
        upsampled_fs: Optional[float],
    ) -> Dict:
        """
        Run cardiac detection on a single signal using template matching.

        Args:
            signal_data (np.ndarray): The input signal data.
            upsampled_signal_data (Optional[np.ndarray]): The upsampled signal data.
            fs (float): The sampling frequency in Hz.
            upsampled_fs (Optional[float]): The upsampled sampling frequency in Hz.

        Returns:
            results (Dict): Detection results for phase_0 with processed signal info.
        """
        # Get upsampled data
        if upsampled_signal_data is None:
            upsampled_signal_data = signal_data.copy()
        if upsampled_fs is None:
            upsampled_fs = fs

        # Estimate detections
        detections, sign = self.detections_estimator.run(
            signal_data=signal_data,
            fs=fs,
        )

        results = {
            "phase_0": {
                "detections": np.array([], dtype=int),
                "prominences": np.array([], dtype=float),
                "are_filled": np.array([], dtype=bool),
                "intervals": np.array([], dtype=int).reshape(0, 2),
            },
            "phase_1": None,
            "processed_signal_score": 0.0,
        }

        # Extract dominant shape
        dominant_shape, upsampled_dominant_shape = self.dominant_shape_extractor.run(
            signal_data=signal_data,
            upsampled_signal_data=upsampled_signal_data,
            detections=detections,
            fs=fs,
            upsampled_fs=upsampled_fs,
        )
        if dominant_shape is None or upsampled_dominant_shape is None:
            if self.debug_print:
                print(
                    "⚠️  No dominant shape could be extracted, returning empty result."
                )
            return results

        results = self.template_matcher.run(
            upsampled_signal_data=upsampled_signal_data,
            upsampled_dominant_shape=upsampled_dominant_shape,
            upsampled_fs=upsampled_fs,
            sign=sign,
        )

        all_detections = results["detections"]
        all_prominences = results["prominences"]
        correlation_mask = (
            results["correlations"]
            >= self.dominant_shape_extracting.good_correlation_threshold
        )
        filtered_detections = results["detections"][correlation_mask]
        filtered_prominences = results["prominences"][correlation_mask]
        filtered_shifts = results["shifts"][correlation_mask]
        filtered_detections = filtered_detections - filtered_shifts
        filtered_correlations = results["correlations"][correlation_mask]

        # Get processed signal score
        processed_signal_score_dict = get_processed_signal_score_dict(
            detections=filtered_detections,
            prominences=filtered_prominences,
            prominence_alpha=self.signal_scoring.prominence_alpha,
            interval_alpha=self.signal_scoring.interval_alpha,
            penalty_alpha=self.signal_scoring.penalty_alpha,
        )
        processed_signal_score = processed_signal_score_dict["score"]

        results = {
            "detections": filtered_detections,
            "prominences": filtered_prominences,
        }

        # Fill missing detections
        if self.detection_filling.enable:
            results = self.missing_detection_filler.run(
                results=results,
                fs=upsampled_fs,
                snapping_detections=all_detections,
                snapping_prominences=all_prominences,
            )
        else:
            results["are_filled"] = np.zeros(results["detections"].shape, dtype=bool)

        # Detect intervals
        results = sort_detection_results(results)
        results = self.interval_detector.run(
            results=results,
            fs=upsampled_fs,
        )

        # Plot detections if debug enabled
        if self.debug_plot:
            plot_detections(
                signal_data=upsampled_signal_data,
                results=results,
                fs=upsampled_fs,
            )

        return {
            "phase_0": results,
            "phase_1": None,
            "processed_signal_score": processed_signal_score,
        }

    def run_ensemble(
        self,
        signal_data_list: List[np.ndarray],
        upsampled_signal_data_list: Optional[List[np.ndarray]],
        fs: float,
        upsampled_fs: Optional[float],
    ) -> Tuple[List[Dict], Dict]:
        """
        Run cardiac detection using ensemble method on multiple signals.

        This method performs template matching on each signal independently using
        per-signal dominant shapes, then synchronizes detections across signals
        using ensemble likelihood estimation. All processing occurs in the upsampled
        domain for consistency.

        Args:
            signal_data_list (List[np.ndarray]): List of signal data arrays.
            upsampled_signal_data_list (Optional[List[np.ndarray]]): List of upsampled signal data arrays.
            fs (float): The sampling frequency in Hz.
            upsampled_fs (Optional[float]): The upsampled sampling frequency in Hz.

        Returns:
            signal_results_list (List[Dict]): Individual signal results.
            ensemble_results (Dict): Ensemble results.

        Raises:
            ValueError: If signal data list is empty or sampling frequency is invalid.
        """
        if len(signal_data_list) == 0:
            raise ValueError("❌ Signal data list is empty.")
        if fs <= 0:
            raise ValueError("❌ Sampling frequency must be positive.")

        # Handle upsampling defaults
        if upsampled_signal_data_list is None:
            upsampled_signal_data_list = [sig.copy() for sig in signal_data_list]
        if upsampled_fs is None:
            upsampled_fs = fs

        # Run detection on each signal separately
        # Save debug flags
        save_debug_print = self.debug_print
        save_debug_plot = self.debug_plot

        # Suppress all debug flags for batch processing
        self.debug_print = False
        self.debug_plot = False
        self.detections_estimator.debug_plot = False
        self.dominant_shape_extractor.debug_print = False
        self.dominant_shape_extractor.debug_plot = False
        self.dominant_shape_extractor.similarity_clusterer.debug_print = False
        self.dominant_shape_extractor.similarity_clusterer.debug_plot = False
        self.template_matcher.debug_print = False
        self.template_matcher.debug_plot = False
        self.template_matcher.similarity_analyzer.debug_print = False
        self.template_matcher.similarity_analyzer.debug_plot = False
        self.missing_detection_filler.debug_print = False
        self.interval_detector.debug_print = False

        signal_results_list = []
        for signal_data, upsampled_signal_data in zip(
            signal_data_list, upsampled_signal_data_list
        ):
            signal_results = self.run(
                signal_data=signal_data,
                upsampled_signal_data=upsampled_signal_data,
                fs=fs,
                upsampled_fs=upsampled_fs,
            )
            # Only include signals with valid phase_0 results
            if signal_results["phase_0"] is not None:
                signal_results_list.append(signal_results)

        # Restore debug flags
        self.debug_print = save_debug_print
        self.debug_plot = save_debug_plot
        self.detections_estimator.debug_plot = save_debug_plot
        self.dominant_shape_extractor.debug_print = save_debug_print
        self.dominant_shape_extractor.debug_plot = save_debug_plot
        self.dominant_shape_extractor.similarity_clusterer.debug_print = (
            save_debug_print
        )
        self.dominant_shape_extractor.similarity_clusterer.debug_plot = save_debug_plot
        self.template_matcher.debug_print = save_debug_print
        self.template_matcher.debug_plot = save_debug_plot
        self.template_matcher.similarity_analyzer.debug_print = save_debug_print
        self.template_matcher.similarity_analyzer.debug_plot = save_debug_plot
        self.missing_detection_filler.debug_print = save_debug_print
        self.interval_detector.debug_print = save_debug_print

        # Process ensemble
        ensemble_results = self._process_ensemble(
            signal_results_list=signal_results_list,
            upsampled_signal_data_list=upsampled_signal_data_list,
            upsampled_fs=upsampled_fs,
        )

        return signal_results_list, ensemble_results

    def run_chunked(
        self,
        signal_data: np.ndarray,
        upsampled_signal_data: Optional[np.ndarray],
        fs: float,
        upsampled_fs: Optional[float],
        processing_regimes: Optional[List[Regime]] = None,
        noise_regimes: Optional[List[Regime]] = None,
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, Dict, np.ndarray]:
        """
        Run cardiac detection on a signal using chunked processing.

        Args:
            signal_data (np.ndarray): The input signal data.
            upsampled_signal_data (Optional[np.ndarray]): The upsampled signal data.
            fs (float): The sampling frequency in Hz.
            upsampled_fs (Optional[float]): The upsampled sampling frequency in Hz.
            processing_regimes (Optional[List[Regime]]): Pre-computed processing regimes. Defaults to None.
            noise_regimes (Optional[List[Regime]]): Pre-computed noise regimes. Defaults to None.

        Returns:
            chunked_results_list (List[Dict]): Chunked results.
            merged_chunked_results (Dict): Merged results.
            range_indices (np.ndarray): Range indices.
        """
        # Handle upsampling defaults
        if upsampled_signal_data is None:
            upsampled_signal_data = signal_data.copy()
        if upsampled_fs is None:
            upsampled_fs = fs

        if self.regime_detection.enable:
            # Detect processing regimes if not provided
            if processing_regimes is None or noise_regimes is None:
                if self.debug_print:
                    print(
                        "⚙️  Regime detection is enabled and no regimes provided. Detecting processing regimes for chunking..."
                    )
                processing_regimes, noise_regimes = self.regime_detector.run(
                    signal_data_list=[signal_data],
                    fs=fs,
                )

            # Chunk signal based on detected processing regimes
            # Use upsampled signal for all processing
            chunk_params = []
            start_indices = np.array(
                [processing_regime.start for processing_regime in processing_regimes],
                dtype=int,
            )
            stop_indices = np.array(
                [processing_regime.stop for processing_regime in processing_regimes],
                dtype=int,
            )

            # Scale indices to upsampled domain
            upsampling_ratio = upsampled_fs / fs
            upsampled_start_indices = (start_indices * upsampling_ratio).astype(int)
            upsampled_stop_indices = (stop_indices * upsampling_ratio).astype(int)

            for processing_regime, upsamp_start, upsamp_stop in zip(
                processing_regimes, upsampled_start_indices, upsampled_stop_indices
            ):
                start_idx = processing_regime.start
                stop_idx = processing_regime.stop
                chunk_data = signal_data[start_idx:stop_idx]
                upsampled_chunk_data = upsampled_signal_data[upsamp_start:upsamp_stop]
                chunk_params.append(
                    (chunk_data, upsampled_chunk_data, fs, upsampled_fs)
                )

            # Create noise mask in upsampled domain
            noise_mask = np.ones(len(upsampled_signal_data), dtype=bool)
            for noise_regime in noise_regimes:
                upsamp_noise_start = int(noise_regime.start * upsampling_ratio)
                upsamp_noise_stop = int(noise_regime.stop * upsampling_ratio)
                noise_mask[upsamp_noise_start:upsamp_noise_stop] = False
        else:
            # Chunk entire signal uniformly
            chunk_size = int(self.chunk_processing.chunk_size_s * fs)
            step_size = int(chunk_size * (1.0 - self.chunk_processing.overlap_ratio))
            upsampled_chunk_size = int(
                self.chunk_processing.chunk_size_s * upsampled_fs
            )
            upsampled_step_size = int(
                upsampled_chunk_size * (1.0 - self.chunk_processing.overlap_ratio)
            )

            if self.debug_print:
                print(
                    "⚠️  Regime detection is disabled. Running chunked detection on entire signal."
                )
            chunk_params = []
            start_indices = np.arange(0, len(signal_data), step_size)
            upsampled_start_indices = np.arange(
                0, len(upsampled_signal_data), upsampled_step_size
            )
            stop_indices = []
            upsampled_stop_indices = []
            for start_idx, upsamp_start_idx in zip(
                start_indices, upsampled_start_indices
            ):
                stop_idx = min(start_idx + chunk_size, len(signal_data))
                upsamp_stop_idx = min(
                    upsamp_start_idx + upsampled_chunk_size, len(upsampled_signal_data)
                )
                chunk_data = signal_data[start_idx:stop_idx]
                upsampled_chunk_data = upsampled_signal_data[
                    upsamp_start_idx:upsamp_stop_idx
                ]
                chunk_params.append(
                    (chunk_data, upsampled_chunk_data, fs, upsampled_fs)
                )
                stop_indices.append(stop_idx)
                upsampled_stop_indices.append(upsamp_stop_idx)

            start_indices = start_indices.astype(int)
            stop_indices = np.array(stop_indices, dtype=int)
            upsampled_start_indices = upsampled_start_indices.astype(int)
            upsampled_stop_indices = np.array(upsampled_stop_indices, dtype=int)

            # Default noiseless mask
            noise_mask = np.ones(len(upsampled_signal_data), dtype=bool)

        # Process chunks
        if self.chunk_processing.parallelize:
            ExecutorClass = (
                ProcessPoolExecutor
                if self.chunk_processing.use_processes
                else ThreadPoolExecutor
            )
            with ExecutorClass(max_workers=self.chunk_processing.max_workers) as executor:
                future_to_params = {
                    executor.submit(
                        self.run, chunk_data, upsampled_chunk_data, fs, upsampled_fs
                    ): i
                    for i, (
                        chunk_data,
                        upsampled_chunk_data,
                        fs,
                        upsampled_fs,
                    ) in enumerate(chunk_params)
                }

                # Collect results maintaining order
                chunked_results_list = [None] * len(chunk_params)
                iterable = future_to_params
                if show_progress:
                    iterable = tqdm(
                        iterable,
                        desc="⌛ Processing chunks in parallel...",
                        position=2,
                        leave=False,
                    )
                for future in iterable:
                    chunk_idx = future_to_params[future]
                    chunked_results = future.result()
                    chunked_results_list[chunk_idx] = chunked_results
        else:
            chunked_results_list = []
            iterable = chunk_params
            if show_progress:
                iterable = tqdm(
                    iterable,
                    desc="⌛ Processing chunks sequentially...",
                    position=2,
                    leave=False,
                )
            for chunk_data, upsampled_chunk_data, fs, upsampled_fs in iterable:
                chunked_results = self.run(
                    chunk_data, upsampled_chunk_data, fs, upsampled_fs
                )
                chunked_results_list.append(chunked_results)

        # Merge chunked results
        merged_chunked_results = self._merge_chunked_results_list(
            chunked_results_list=chunked_results_list,
            start_indices=upsampled_start_indices,
            stop_indices=upsampled_stop_indices,
            fs=upsampled_fs,
        )

        # Apply noise mask to merged results
        phase_results = merged_chunked_results["phase_0"]
        if phase_results is not None and len(phase_results["detections"]) > 0:
            valid_detections_mask = noise_mask[phase_results["detections"]]

            # Remap intervals to account for removed detections
            old_to_new = np.full(len(valid_detections_mask), -1, dtype=int)
            old_to_new[valid_detections_mask] = np.arange(valid_detections_mask.sum())

            # Filter intervals where both start and end are valid
            intervals = phase_results["intervals"]
            if len(intervals) == 0:
                remapped_intervals = np.array([], dtype=int).reshape(0, 2)
            else:
                valid_intervals_mask = (
                    valid_detections_mask[intervals[:, 0]]
                    & valid_detections_mask[intervals[:, 1]]
                )
                valid_intervals = intervals[valid_intervals_mask]
                remapped_intervals = old_to_new[valid_intervals]

            merged_chunked_results["phase_0"] = {
                "detections": phase_results["detections"][valid_detections_mask],
                "prominences": phase_results["prominences"][valid_detections_mask],
                "are_filled": phase_results["are_filled"][valid_detections_mask],
                "intervals": remapped_intervals,
            }

        # Return indices in upsampled domain
        range_indices = np.vstack((upsampled_start_indices, upsampled_stop_indices)).T

        return chunked_results_list, merged_chunked_results, range_indices

    def run_ensemble_chunked(
        self,
        signal_data_list: List[np.ndarray],
        upsampled_signal_data_list: Optional[List[np.ndarray]],
        fs: float,
        upsampled_fs: Optional[float],
    ) -> Tuple[List[Dict], Dict]:
        """
        Run ensemble cardiac detection on multiple signals using chunked processing.

        Args:
            signal_data_list (List[np.ndarray]): List of signal data arrays.
            upsampled_signal_data_list (Optional[List[np.ndarray]]): List of upsampled signal data arrays.
            fs (float): The sampling frequency in Hz.
            upsampled_fs (Optional[float]): The upsampled sampling frequency in Hz.

        Returns:
            individual_merged_chunked_results_list (List[Dict]): Individual merged results.
            merged_ensemble_chunked_results (Dict): Ensemble merged results.

        Raises:
            ValueError: If signal data list is empty, sampling frequency is invalid, or signals have different lengths.
        """
        # Validate inputs
        if len(signal_data_list) == 0:
            raise ValueError("❌ Signal data list is empty.")
        if fs <= 0:
            raise ValueError("❌ Sampling frequency must be positive.")

        # Handle upsampling defaults
        if upsampled_signal_data_list is None:
            upsampled_signal_data_list = [sig.copy() for sig in signal_data_list]
        if upsampled_fs is None:
            upsampled_fs = fs

        # Validate all signals have same length
        signal_length = len(signal_data_list[0])
        for i, signal_data in enumerate(signal_data_list[1:], 1):
            if len(signal_data) != signal_length:
                raise ValueError(
                    f"❌ Signal {i} has different length ({len(signal_data)}) "
                    f"than first signal ({signal_length})."
                )

        # Validate all upsampled signals have same length
        upsampled_signal_length = len(upsampled_signal_data_list[0])
        for i, upsampled_signal_data in enumerate(upsampled_signal_data_list[1:], 1):
            if len(upsampled_signal_data) != upsampled_signal_length:
                raise ValueError(
                    f"❌ Upsampled signal {i} has different length ({len(upsampled_signal_data)}) "
                    f"than first upsampled signal ({upsampled_signal_length})."
                )

        # Compute regimes from signals
        if self.regime_detection.enable:
            if self.debug_print:
                print(
                    "⚙️  Regime detection is enabled. Detecting processing regimes for chunking..."
                )
            processing_regimes, noise_regimes = self.regime_detector.run(
                signal_data_list=signal_data_list,
                fs=fs,
            )
        else:
            processing_regimes = None
            noise_regimes = None

        # Run chunked detection on each signal
        if self.sensor_processing.parallelize:
            ExecutorClass = (
                ProcessPoolExecutor
                if self.sensor_processing.use_processes
                else ThreadPoolExecutor
            )
            n_signals = len(signal_data_list)
            with ExecutorClass(max_workers=self.sensor_processing.max_workers) as executor:
                future_to_idx = {
                    executor.submit(
                        self.run_chunked,
                        signal_data=signal_data,
                        upsampled_signal_data=upsampled_signal_data,
                        fs=fs,
                        upsampled_fs=upsampled_fs,
                        processing_regimes=processing_regimes,
                        noise_regimes=noise_regimes,
                        show_progress=(i == 0),
                    ): i
                    for i, (signal_data, upsampled_signal_data) in enumerate(
                        zip(signal_data_list, upsampled_signal_data_list)
                    )
                }

                # Collect results maintaining order
                ordered_results = [None] * n_signals
                for future in tqdm(
                    future_to_idx,
                    desc="⌛ Processing individual signals in parallel...",
                    position=1,
                    leave=False,
                ):
                    idx = future_to_idx[future]
                    ordered_results[idx] = future.result()

            individual_merged_chunked_results_list = [r[1] for r in ordered_results]
            individual_chunked_results_list = [r[0] for r in ordered_results]
            range_indices_list = [r[2] for r in ordered_results]
        else:
            individual_merged_chunked_results_list = []
            individual_chunked_results_list = []
            range_indices_list = []
            for signal_data, upsampled_signal_data in tqdm(
                zip(signal_data_list, upsampled_signal_data_list),
                desc="⌛ Processing individual signals sequentially...",
                position=1,
                leave=False,
            ):
                chunked_results_list, merged_chunked_results, range_indices = (
                    self.run_chunked(
                        signal_data=signal_data,
                        upsampled_signal_data=upsampled_signal_data,
                        fs=fs,
                        upsampled_fs=upsampled_fs,
                        processing_regimes=processing_regimes,
                        noise_regimes=noise_regimes,
                    )
                )
                individual_merged_chunked_results_list.append(merged_chunked_results)
                individual_chunked_results_list.append(chunked_results_list)
                range_indices_list.append(range_indices)

        # Check that all signals have the same number of chunks
        n_chunks = len(individual_chunked_results_list[0])
        for j in range(1, len(signal_data_list)):
            if len(individual_chunked_results_list[j]) != n_chunks:
                raise ValueError(
                    "❌ Signals have different number of chunks, cannot perform ensemble detection."
                )

        # Check that all start indices are the same
        for range_indices in range_indices_list[1:]:
            if not np.array_equal(range_indices_list[0], range_indices):
                raise ValueError(
                    "❌ Signals have different chunk start indices, cannot perform ensemble detection."
                )

        # Process ensemble for each chunk
        ensemble_chunked_results_list = []
        for i in tqdm(
            range(n_chunks),
            desc="⏳ Processing ensemble chunks...",
            position=1,
            leave=False,
        ):
            chunked_results_list = [
                individual_chunked_results_list[j][i]
                for j in range(len(signal_data_list))
            ]

            chunked_upsampled_signal_data_list = []
            for j, upsampled_signal_data in enumerate(upsampled_signal_data_list):
                start_idx = range_indices_list[j][i][0]
                stop_idx = range_indices_list[j][i][1]
                chunked_upsampled_signal_data = upsampled_signal_data[
                    start_idx:stop_idx
                ]
                chunked_upsampled_signal_data_list.append(chunked_upsampled_signal_data)

            ensemble_chunked_results = self._process_ensemble(
                signal_results_list=chunked_results_list,
                upsampled_signal_data_list=chunked_upsampled_signal_data_list,
                upsampled_fs=upsampled_fs,
            )
            ensemble_chunked_results_list.append(ensemble_chunked_results)

        # Merge ensemble results
        start_indices = range_indices_list[0][:, 0]
        stop_indices = range_indices_list[0][:, 1]
        merged_ensemble_chunked_results = self._merge_chunked_results_list(
            chunked_results_list=ensemble_chunked_results_list,
            start_indices=start_indices,
            stop_indices=stop_indices,
            fs=upsampled_fs,
        )

        return individual_merged_chunked_results_list, merged_ensemble_chunked_results