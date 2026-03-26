"""
A class for detecting respiratory events from physiological signals.

This module implements the RespiratoryDetector class which uses adaptive ensemble
methods to detect respiratory cycles from multi-channel physiological signals.
It supports regime-based chunked processing for long recordings and includes
signal preprocessing, peak detection, interval detection, and ensemble fusion.

Authors: Arnaud Poletto
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib.patches import Patch
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from cinc.utils.signal_processor import SignalProcessor
from cinc.core.detection.utils.plot import plot_detections
from cinc.utils.config import dict_to_namespace, load_config
from cinc.core.detection.utils.cardiac_detection.signal_score import (
    get_processed_signal_score_dict,
)
from cinc.core.detection.utils.sort_results import sort_detection_results
from cinc.core.detection.utils.regime_detector import Regime, RegimeDetector
from cinc.core.detection.utils.ensemble.ensemble_likelihood_estimator import (
    EnsembleLikelihoodEstimator,
)
from cinc.core.detection.utils.interval.interval_detector import IntervalDetector
from cinc.core.detection.utils.ensemble.ensemble_detector import EnsembleDetector
from cinc.core.detection.utils.fill.missing_detection_filler import (
    MissingDetectionFiller,
)
from cinc.core.detection.utils.respiratory_detection.intervals.interval_processor import (
    IntervalProcessor,
)


class RespiratoryDetector:
    """A class for detecting respiratory events from physiological signals using adaptive ensemble methods."""

    def __init__(
        self,
        config_file_path: str,
        detect_both_phases: bool = False,
        debug_plot: bool = False,
        debug_print: bool = False,
    ) -> None:
        """
        Initialize the respiratory detector with configuration and debug settings.

        Args:
            config_file_path (str): Path to the YAML configuration file.
            detect_both_phases (bool): Whether to detect both inspiration and expiration. Defaults to False.
            debug_plot (bool): Whether to enable debug plotting. Defaults to False.
            debug_print (bool): Whether to enable debug printing. Defaults to False.
        """
        super(RespiratoryDetector, self).__init__()

        self.detect_both_phases = detect_both_phases
        self.debug_plot = debug_plot
        self.debug_print = debug_print

        # Load configuration and convert to namespace for attribute access
        config = load_config(config_file_path)
        for parameter, value in config.items():
            setattr(self, parameter, dict_to_namespace(value))
        self.config = config

        # Initialize missing detection filler
        signal_missing_detection_filler_config = {
            "interval_estimation_window_s": self.detection_filling.interval_estimation_window_s,
            "gap_detection_multiplier": self.detection_filling.gap_detection_multiplier,
            "max_fill_count": self.detection_filling.max_fill_count,
            "min_snap_distance_s": self.detection_filling.min_snap_distance_s,
        }
        self.signal_missing_detection_filler = MissingDetectionFiller(
            config=signal_missing_detection_filler_config,
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
            cardiac_mode=False,
            debug_print=debug_print,
        )

        # Initialize interval processor
        self.interval_processor = IntervalProcessor()

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
            detect_both_phases=detect_both_phases,
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
            detect_both_phases=detect_both_phases,
            cardiac_mode=False,
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

    def _get_processed_signals(
        self,
        signal_data: np.ndarray,
        fs: float,
    ) -> List[np.ndarray]:
        """
        Generate multiple bandpass-filtered versions of the signal with different cutoff frequencies.

        Args:
            signal_data (np.ndarray): The input signal data.
            fs (float): The sampling frequency in Hz.

        Returns:
            processed_signals_data (List[np.ndarray]): List of processed signals with different highcut frequencies.
        """
        return [
            SignalProcessor.bandpass_filter(
                signal_data=signal_data,
                lowcut=self.signal_processing.signal_lowcut_fs,
                highcut=highcut_fs,
                fs=fs,
                order=3,
            )
            for highcut_fs in self.signal_processing.signal_highcut_fss
        ]

    def _get_best_processed_signal(
        self,
        signal_data: np.ndarray,
        processed_signals_data: List[np.ndarray],
        fs: float,
    ) -> Tuple[np.ndarray, float]:
        """
        Select the best processed signal based on quality scores.

        This method evaluates each processed signal variant using prominence,
        interval consistency, and penalty metrics to select the optimal filter configuration.

        Args:
            signal_data (np.ndarray): The original signal data.
            processed_signals_data (List[np.ndarray]): List of processed signal variants.
            fs (float): The sampling frequency in Hz.

        Returns:
            processed_signal_data (np.ndarray): Best processed signal.
            processed_signal_score (float): Quality score of the best processed signal.
        """
        prominence_scores = []
        interval_scores = []
        penalty_scores = []
        scores = []
        for processed_signal_data, highcut_fs in zip(
            processed_signals_data, self.signal_processing.signal_highcut_fss
        ):
            # Calculate detection parameters
            distance = int(0.5 / highcut_fs * fs)
            prominence = (
                np.std(processed_signal_data)
                * self.peak_detection.prominence_std_factor
                if self.peak_detection.prominence_std_factor is not None
                else 0
            )

            # Get positive phase detections
            pos_detections, pos_properties = find_peaks(
                processed_signal_data,
                distance=distance,
                prominence=prominence,
            )
            pos_prominences = pos_properties["prominences"]
            pos_score_dict = get_processed_signal_score_dict(
                detections=pos_detections,
                prominences=pos_prominences,
                prominence_alpha=self.signal_scoring.prominence_alpha,
                interval_alpha=self.signal_scoring.interval_alpha,
                penalty_alpha=self.signal_scoring.penalty_alpha,
            )

            # Get negative phase detections if needed
            if self.detect_both_phases:
                neg_detections, neg_properties = find_peaks(
                    -processed_signal_data,
                    distance=distance,
                    prominence=prominence,
                )
                neg_prominences = neg_properties["prominences"]
                neg_score_dict = get_processed_signal_score_dict(
                    detections=neg_detections,
                    prominences=neg_prominences,
                    prominence_alpha=self.signal_scoring.prominence_alpha,
                    interval_alpha=self.signal_scoring.interval_alpha,
                    penalty_alpha=self.signal_scoring.penalty_alpha,
                )

            # Combine scores from both phases
            if self.detect_both_phases:
                prominence_score = (
                    pos_score_dict["prominence_score"]
                    + neg_score_dict["prominence_score"]
                ) / 2
                interval_score = (
                    pos_score_dict["interval_score"] + neg_score_dict["interval_score"]
                ) / 2
                penalty_score = (
                    pos_score_dict["penalty_score"] + neg_score_dict["penalty_score"]
                ) / 2
                score = (pos_score_dict["score"] + neg_score_dict["score"]) / 2
            else:
                prominence_score = pos_score_dict["prominence_score"]
                interval_score = pos_score_dict["interval_score"]
                penalty_score = pos_score_dict["penalty_score"]
                score = pos_score_dict["score"]

            prominence_scores.append(prominence_score)
            interval_scores.append(interval_score)
            penalty_scores.append(penalty_score)
            scores.append(score)

        # Handle case where all scores are NaN
        if all(np.isnan(scores)):
            if self.debug_print:
                print(
                    "⚠️  All processed signals have NaN scores. Returning the first processed signal."
                )
            return processed_signals_data[0], float("nan")

        best_index = np.nanargmin(scores)
        best_score = scores[best_index]

        # Plot comparison if debug enabled
        if self.debug_plot:
            self._plot_best_processed_signal(
                signal_data=signal_data,
                processed_signals_data=processed_signals_data,
                prominence_scores=prominence_scores,
                interval_scores=interval_scores,
                penalty_scores=penalty_scores,
                scores=scores,
                best_index=best_index,
                fs=fs,
            )

        if self.debug_print:
            print(
                f"✅ Selected processed frequency {self.signal_processing.signal_highcut_fss[best_index]} Hz with score {best_score:.4f}."
            )

        return processed_signals_data[best_index], best_score

    def _get_results(
        self,
        processed_signal_data: np.ndarray,
        fs: float,
    ) -> Dict:
        """
        Extract respiratory events from a processed signal.

        This method performs peak detection, fills missing detections,
        filters short intervals, and detects respiratory intervals.

        Args:
            processed_signal_data (np.ndarray): The processed signal data.
            fs (float): The sampling frequency in Hz.

        Returns:
            results (Dict): Detection results containing detections, prominences, filled flags, and intervals.
        """
        distance = int(self.interval_detection.min_interval_s * fs)

        # Get all detections without prominence filtering for snapping
        all_detections, all_properties = find_peaks(
            processed_signal_data,
            distance=distance,
            prominence=0,
        )
        all_prominences = all_properties["prominences"]

        # Get detections with prominence filtering for final results
        prominence = (
            np.std(processed_signal_data) * self.peak_detection.prominence_std_factor
            if self.peak_detection.prominence_std_factor is not None
            else 0
        )
        detections, properties = find_peaks(
            processed_signal_data,
            distance=distance,
            prominence=prominence,
        )
        prominences = properties["prominences"]

        results = {
            "detections": detections,
            "prominences": prominences,
        }

        # Fill missing detections
        if self.detection_filling.enable:
            results = self.signal_missing_detection_filler.run(
                results=results,
                fs=fs,
                snapping_detections=all_detections,
                snapping_prominences=all_prominences,
            )
        else:
            results["are_filled"] = np.zeros(results["detections"].shape, dtype=bool)

        # Detect intervals
        results = sort_detection_results(results)
        results = self.interval_detector.run(
            results=results,
            fs=fs,
        )

        return results

    def _process_ensemble(
        self,
        signal_results_list: List[Dict],
        signal_data_list: List[np.ndarray],
        fs: float,
    ) -> Dict:
        """
        Combine individual signal results using ensemble detection.

        Args:
            signal_results_list (List[Dict]): List of detection results from individual signals.
            signal_data_list (List[np.ndarray]): List of signal data arrays.
            fs (float): The sampling frequency in Hz.

        Returns:
            ensemble_results (Dict): Ensemble detection results for both phases with signs.
        """
        # Compute likelihood and global parameters
        likelihood, global_median_interval_length_s, signs = (
            self.ensemble_likelihood_estimator.run(
                results_list=signal_results_list,
                signal_length=signal_data_list[0].shape[0],
                fs=fs,
            )
        )

        if (
            likelihood is None
            or global_median_interval_length_s is None
            or signs is None
        ):
            if self.debug_print:
                print(
                    "⚠️  Cannot compute ensemble detection without likelihood or global median interval length."
                )

            return {
                "phase_0": {
                    "detections": np.array([], dtype=int),
                    "prominences": np.array([], dtype=float),
                    "are_filled": np.array([], dtype=bool),
                    "intervals": np.array([], dtype=int).reshape(0, 2),
                },
                "phase_1": {
                    "detections": np.array([], dtype=int),
                    "prominences": np.array([], dtype=float),
                    "are_filled": np.array([], dtype=bool),
                    "intervals": np.array([], dtype=int).reshape(0, 2),
                }
                if self.detect_both_phases
                else None,
                "signs": np.zeros(len(signal_results_list), dtype=int),
            }

        # Run ensemble detection
        ensemble_results = self.ensemble_detector.run(
            likelihood=likelihood,
            global_median_interval_length_s=global_median_interval_length_s,
            fs=fs,
        )
        ensemble_results["signs"] = signs

        # Plot detections if debug enabled
        if self.debug_plot:
            if self.detect_both_phases:
                plot_detections(
                    signal_data=likelihood,
                    results=[ensemble_results["phase_0"], ensemble_results["phase_1"]],
                    fs=fs,
                )
            else:
                plot_detections(
                    signal_data=likelihood,
                    results=ensemble_results["phase_0"],
                    fs=fs,
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

            phase_1_results = ensemble_results["phase_1"]
            if phase_1_results is not None:
                print("✅ Phase 1 results with keys:")
                for key in phase_1_results:
                    shape_str = (
                        f", shape: {phase_1_results[key].shape}"
                        if isinstance(phase_1_results[key], np.ndarray)
                        else ""
                    )
                    print(
                        f"\t- {key} (phase 1): {type(phase_1_results[key])}{shape_str}"
                    )

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

        Args:
            chunked_results_list (List[Dict]): List of detection results from individual chunks.
            start_indices (np.ndarray): Array of start indices for each chunk.
            stop_indices (np.ndarray): Array of stop indices for each chunk.
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
            "phase_1": {
                "detections": np.array([], dtype=int),
                "prominences": np.array([], dtype=float),
                "are_filled": np.array([], dtype=bool),
            }
            if self.detect_both_phases
            else None,
        }

        for chunked_results, start_idx, stop_idx in zip(
            chunked_results_list, start_indices, stop_indices
        ):
            for phase_key in ["phase_0", "phase_1"]:
                phase_results = chunked_results[phase_key]

                # Validate phase results existence
                if phase_results is None:
                    if self.debug_print:
                        print(
                            f"⚠️  Chunk at index {start_idx}-{stop_idx} has no results for {phase_key}, skipping."
                        )
                    continue

                # Validate phase results required keys
                required_keys = ["detections", "prominences", "are_filled"]
                if not all(key in phase_results for key in required_keys):
                    if self.debug_print:
                        print(
                            f"⚠️  Chunk at index {start_idx}-{stop_idx} missing required keys for {phase_key}, skipping."
                        )
                    continue

                # Merge phase results
                merged_chunked_results[phase_key] = {
                    "detections": np.concatenate(
                        [
                            merged_chunked_results[phase_key]["detections"],
                            phase_results["detections"]
                            + start_idx,  # Offset by chunk start index
                        ]
                    ),
                    "prominences": np.concatenate(
                        [
                            merged_chunked_results[phase_key]["prominences"],
                            phase_results["prominences"],
                        ]
                    ),
                    "are_filled": np.concatenate(
                        [
                            merged_chunked_results[phase_key]["are_filled"],
                            phase_results["are_filled"],
                        ]
                    ),
                }

        # Recompute intervals after merging all chunks
        for phase_key in ["phase_0", "phase_1"]:
            if merged_chunked_results[phase_key] is None:
                continue
            merged_chunked_results[phase_key] = sort_detection_results(
                merged_chunked_results[phase_key]
            )
            merged_chunked_results[phase_key] = self.interval_detector.run(
                results=merged_chunked_results[phase_key],
                fs=fs,
            )

        # Append processed signal score if available
        chunks_data = {
            "starts": start_indices,
            "stops": stop_indices,
            "signs_list": [
                chunked_results["signs"] if "signs" in chunked_results else None
                for chunked_results in chunked_results_list
            ],
            "processed_signal_scores": [
                chunked_results["processed_signal_score"]
                if "processed_signal_score" in chunked_results
                else None
                for chunked_results in chunked_results_list
            ],
        }

        # Remove signs or processed signal scores if all are None
        if all(sign is None for sign in chunks_data["signs_list"]):
            chunks_data.pop("signs_list")
        if all(score is None for score in chunks_data["processed_signal_scores"]):
            chunks_data.pop("processed_signal_scores")

        merged_chunked_results["chunks"] = chunks_data

        return merged_chunked_results

    def run(
        self,
        signal_data: np.ndarray,
        fs: float,
    ) -> Dict:
        """
        Run respiratory detection on a single signal.

        Args:
            signal_data (np.ndarray): The input signal data.
            fs (float): The sampling frequency in Hz.

        Returns:
            results (Dict): Detection results for both phases with processed signal info.
        """
        # Get best processed signal
        processed_signals_data = self._get_processed_signals(
            signal_data=signal_data,
            fs=fs,
        )
        processed_signal_data, processed_signal_score = self._get_best_processed_signal(
            signal_data=signal_data,
            processed_signals_data=processed_signals_data,
            fs=fs,
        )

        # Get positive detections
        pos_results = self._get_results(
            processed_signal_data=processed_signal_data,
            fs=fs,
        )

        # Get negative detections if needed
        if self.detect_both_phases:
            neg_results = self._get_results(
                processed_signal_data=-processed_signal_data,
                fs=fs,
            )
        else:
            neg_results = None

        # Plot detections if debug enabled
        if self.debug_plot:
            if self.detect_both_phases:
                plot_detections(
                    signal_data=processed_signal_data,
                    results=[pos_results, neg_results],
                    fs=fs,
                )
            else:
                plot_detections(
                    signal_data=processed_signal_data,
                    results=pos_results,
                    fs=fs,
                )

        results = {
            "phase_0": pos_results,
            "phase_1": neg_results,
            "processed_signal_score": processed_signal_score,
        }

        results = self.interval_processor.run(
            results=results,
        )

        return results

    def run_ensemble(
        self,
        signal_data_list: List[np.ndarray],
        fs: float,
    ) -> Tuple[List[Dict], Dict]:
        """
        Run respiratory detection using ensemble method on multiple signals.

        Args:
            signal_data_list (List[np.ndarray]): List of signal data arrays.
            fs (float): The sampling frequency in Hz.

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

        # Run detection on each signal separately
        save_debug_print = self.debug_print
        save_debug_plot = self.debug_plot
        self.debug_print = False
        self.debug_plot = False
        self.signal_missing_detection_filler.debug_print = False
        self.interval_detector.debug_print = False

        signal_results_list = []
        for signal_data in signal_data_list:
            signal_results = self.run(
                signal_data=signal_data,
                fs=fs,
            )
            # Only include signals with valid phase_0 results
            if signal_results["phase_0"] is not None:
                signal_results_list.append(signal_results)

        self.debug_print = save_debug_print
        self.debug_plot = save_debug_plot
        self.signal_missing_detection_filler.debug_print = save_debug_print
        self.interval_detector.debug_print = save_debug_print

        ensemble_results = self._process_ensemble(
            signal_results_list=signal_results_list,
            signal_data_list=signal_data_list,
            fs=fs,
        )

        return signal_results_list, ensemble_results

    def run_chunked(
        self,
        signal_data: np.ndarray,
        fs: float,
        processing_regimes: Optional[List[Regime]] = None,
        noise_regimes: Optional[List[Regime]] = None,
        show_progress: bool = True,
    ) -> Tuple[List[Dict], Dict, np.ndarray]:
        """
        Run respiratory detection on a signal using chunked processing.

        Args:
            signal_data (np.ndarray): The input signal data.
            fs (float): The sampling frequency in Hz.
            processing_regimes (Optional[List[Regime]]): Pre-computed processing regimes. Defaults to None.
            noise_regimes (Optional[List[Regime]]): Pre-computed noise regimes. Defaults to None.

        Returns:
            chunked_results_list (List[Dict]): Chunked results.
            merged_chunked_results (Dict): Merged results.
            range_indices (np.ndarray): Range indices.
        """
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
            chunk_params = []
            start_indices = np.array(
                [processing_regime.start for processing_regime in processing_regimes],
                dtype=int,
            )
            stop_indices = np.array(
                [processing_regime.stop for processing_regime in processing_regimes],
                dtype=int,
            )
            for processing_regime in processing_regimes:
                start_idx = processing_regime.start
                stop_idx = processing_regime.stop
                chunk_data = signal_data[start_idx:stop_idx]
                chunk_params.append((chunk_data, fs))

            # Create noise mask
            noise_mask = np.ones(len(signal_data), dtype=bool)
            for noise_regime in noise_regimes:
                noise_mask[noise_regime.start : noise_regime.stop] = False
        else:
            # Chunk entire signal uniformly
            chunk_size = int(self.chunk_processing.chunk_size_s * fs)
            step_size = int(chunk_size * (1.0 - self.chunk_processing.overlap_ratio))

            if self.debug_print:
                print(
                    "⚠️  Regime detection is disabled. Running chunked detection on entire signal."
                )
            chunk_params = []
            start_indices = np.arange(0, len(signal_data), step_size)
            stop_indices = []
            for start_idx in start_indices:
                stop_idx = min(start_idx + chunk_size, len(signal_data))
                chunk_data = signal_data[start_idx:stop_idx]
                chunk_params.append((chunk_data, fs))
                stop_indices.append(stop_idx)
            start_indices = start_indices.astype(int)
            stop_indices = np.array(stop_indices, dtype=int)

            # Default noiseless mask
            noise_mask = np.ones(len(signal_data), dtype=bool)

        # Process chunks
        if self.chunk_processing.parallelize:
            ExecutorClass = (
                ProcessPoolExecutor
                if self.chunk_processing.use_processes
                else ThreadPoolExecutor
            )
            with ExecutorClass(max_workers=self.chunk_processing.max_workers) as executor:
                future_to_params = {
                    executor.submit(self.run, chunk_data, fs): i
                    for i, (chunk_data, fs) in enumerate(chunk_params)
                }

                # Collect results maintaining order
                chunked_results_list = [None] * len(chunk_params)
                iterable = future_to_params
                if show_progress:
                    iterable = tqdm(
                        iterable,
                        desc="⌛ Collecting chunk results...",
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
            for chunk_data, fs in iterable:
                chunked_results = self.run(chunk_data, fs)
                chunked_results_list.append(chunked_results)

        # Merge chunked results
        merged_chunked_results = self._merge_chunked_results_list(
            chunked_results_list=chunked_results_list,
            start_indices=start_indices,
            stop_indices=stop_indices,
            fs=fs,
        )

        # Apply noise mask to merged results
        for phase_key in ["phase_0", "phase_1"]:
            if phase_key not in merged_chunked_results:
                continue
            phase_results = merged_chunked_results[phase_key]
            if phase_results is None:
                continue

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

            merged_chunked_results[phase_key] = {
                "detections": phase_results["detections"][valid_detections_mask],
                "prominences": phase_results["prominences"][valid_detections_mask],
                "are_filled": phase_results["are_filled"][valid_detections_mask],
                "intervals": remapped_intervals,
            }

        range_indices = np.vstack((start_indices, stop_indices)).T

        return chunked_results_list, merged_chunked_results, range_indices

    def run_ensemble_chunked(
        self,
        signal_data_list: List[np.ndarray],
        fs: float,
    ) -> Tuple[List[Dict], Dict]:
        """
        Run ensemble respiratory detection on multiple signals using chunked processing.

        Args:
            signal_data_list (List[np.ndarray]): List of signal data arrays.
            fs (float): The sampling frequency in Hz.

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

        # Validate all signals have same length
        signal_length = len(signal_data_list[0])
        for i, signal_data in enumerate(signal_data_list[1:], 1):
            if len(signal_data) != signal_length:
                raise ValueError(
                    f"❌ Signal {i} has different length ({len(signal_data)}) "
                    f"than first signal ({signal_length})."
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
                        fs=fs,
                        processing_regimes=processing_regimes,
                        noise_regimes=noise_regimes,
                        show_progress=(i == 0),
                    ): i
                    for i, signal_data in enumerate(signal_data_list)
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
            for signal_data in tqdm(
                signal_data_list,
                desc="⌛ Processing individual signals...",
                position=1,
                leave=False,
            ):
                chunked_results_list, merged_chunked_results, range_indices = (
                    self.run_chunked(
                        signal_data=signal_data,
                        fs=fs,
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

            chunked_signal_data_list = []
            for j, signal_data in enumerate(signal_data_list):
                start_idx = range_indices_list[j][i][0]
                stop_idx = range_indices_list[j][i][1]
                chunked_signal_data = signal_data[start_idx:stop_idx]
                chunked_signal_data_list.append(chunked_signal_data)

            ensemble_chunked_results = self._process_ensemble(
                signal_results_list=chunked_results_list,
                signal_data_list=chunked_signal_data_list,
                fs=fs,
            )
            ensemble_chunked_results_list.append(ensemble_chunked_results)

        # Merge ensemble results
        start_indices = range_indices_list[0][:, 0]
        stop_indices = range_indices_list[0][:, 1]
        merged_ensemble_chunked_results = self._merge_chunked_results_list(
            chunked_results_list=ensemble_chunked_results_list,
            start_indices=start_indices,
            stop_indices=stop_indices,
            fs=fs,
        )

        return individual_merged_chunked_results_list, merged_ensemble_chunked_results

    def _plot_best_processed_signal(
        self,
        signal_data: np.ndarray,
        processed_signals_data: List[np.ndarray],
        prominence_scores: List[float],
        interval_scores: List[float],
        penalty_scores: List[float],
        scores: List[float],
        best_index: int,
        fs: float,
    ) -> None:
        """
        Plot comparison of processed signals with scores and rankings.

        Args:
            signal_data (np.ndarray): The original signal data.
            processed_signals_data (list): List of processed signal variants.
            prominence_scores (list): Prominence scores for each variant.
            interval_scores (list): Interval scores for each variant.
            penalty_scores (list): Penalty scores for each variant.
            scores (list): Total scores for each variant.
            best_index (int): Index of the best processed signal.
            fs (float): The sampling frequency in Hz.
        """
        n_signals = len(processed_signals_data)

        # Time array for processed signals
        time_processed = np.arange(len(processed_signals_data[0])) / fs
        time_raw = np.arange(len(signal_data)) / fs

        # Resample raw signal to match processed signal length if needed
        if len(time_raw) != len(time_processed):
            raw_resampled = np.interp(time_processed, time_raw, signal_data)
        else:
            raw_resampled = signal_data

        # Remove mean from raw signal for better visualization
        raw_resampled = raw_resampled - np.mean(raw_resampled)

        # Calculate ranking
        valid_scores = [(i, s) for i, s in enumerate(scores) if not np.isnan(s)]
        ranked_indices = sorted(valid_scores, key=lambda x: x[1])
        rank_map = {idx: rank + 1 for rank, (idx, _) in enumerate(ranked_indices)}

        # Calculate grid layout
        n_cols = 2
        n_rows_processed = int(np.ceil(n_signals / n_cols))

        # Create figure and gridspec
        fig = plt.figure(figsize=(20, 3 * n_rows_processed + 2.5))
        gs = fig.add_gridspec(
            n_rows_processed + 1,
            n_cols,
            height_ratios=[1] * n_rows_processed + [0.8],
            hspace=0.3,
            wspace=0.25,
        )

        # Plot 1: Processed signals in 2-column grid layout
        axes_processed = []
        for i, (
            processed_signal_data,
            highcut_fs,
            prominence_score,
            interval_score,
            penalty_score,
            score,
        ) in enumerate(
            zip(
                processed_signals_data,
                self.signal_processing.signal_highcut_fss,
                prominence_scores,
                interval_scores,
                penalty_scores,
                scores,
            )
        ):
            # Calculate subplot position
            row = i // n_cols
            col = i % n_cols

            # Create subplot
            if i == 0:
                ax = fig.add_subplot(gs[row, col])
                ax_first = ax
            else:
                ax = fig.add_subplot(gs[row, col], sharex=ax_first)
            axes_processed.append(ax)

            # Visual properties
            is_best = i == best_index
            color = "#2ecc71" if is_best else "#7f8c8d"
            linewidth = 2.5 if is_best else 1.2
            alpha = 1.0 if is_best else 0.85

            # Plot raw signal in background
            ax.plot(
                time_processed,
                raw_resampled,
                color="#bdc3c7",
                linewidth=0.8,
                alpha=0.3,
                zorder=1,
            )

            # Plot processed signal on top
            ax.plot(
                time_processed,
                processed_signal_data,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                zorder=2,
            )

            # Add rank badge in top-left corner
            rank = rank_map.get(i, n_signals)
            badge_color = "#2ecc71" if is_best else "#95a5a6"
            ax.text(
                0.02,
                0.97,
                f"#{rank}",
                transform=ax.transAxes,
                fontsize=14,
                fontweight="bold",
                verticalalignment="top",
                bbox=dict(
                    boxstyle="circle,pad=0.3",
                    facecolor=badge_color,
                    edgecolor="white",
                    linewidth=2,
                    alpha=0.9,
                ),
                color="white",
                zorder=10,
            )

            # Title with frequency and total score
            score_str = f"{score:.3f}" if not np.isinf(score) else "∞"
            best_marker = " ★" if is_best else ""
            ax.set_title(
                f"Highcut: {highcut_fs} Hz | Total Score: {score_str}{best_marker}",
                fontsize=11,
                fontweight="bold" if is_best else "normal",
                color="#2ecc71" if is_best else "#34495e",
                pad=8,
            )

            # Add score component breakdown in top-right corner
            score_text = (
                f"Prom: {prominence_score:.2f}\n"
                f"Intv: {interval_score:.2f}\n"
                f"Pnlt: {penalty_score:.2f}"
            )
            ax.text(
                0.98,
                0.97,
                score_text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="white",
                    edgecolor=color,
                    linewidth=1.5,
                    alpha=0.9,
                ),
                color="#34495e",
                family="monospace",
            )

            # Styling
            ax.set_ylabel("Amplitude", fontsize=9)
            ax.grid(True, alpha=0.2, linestyle="--")
            ax.set_xlim(0, time_processed[-1])

            # Highlight best signal with bold border
            if is_best:
                for spine in ax.spines.values():
                    spine.set_edgecolor("#2ecc71")
                    spine.set_linewidth(3)
            else:
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

            # Only show x-label and ticks on bottom row
            if row == n_rows_processed - 1:
                ax.set_xlabel("Time (s)", fontsize=10)
            else:
                ax.tick_params(labelbottom=False)

        # Plot 2: Score comparison bars
        ax_scores = fig.add_subplot(gs[-1, :])

        # Prepare data for stacked horizontal bars
        valid_indices = [i for i in range(n_signals) if not np.isnan(scores[i])]
        sorted_indices = sorted(
            valid_indices, key=lambda i: self.signal_processing.signal_highcut_fss[i]
        )

        y_pos = np.arange(len(sorted_indices))
        bar_height = 0.7

        # Custom legend handles
        legend_elements = [
            Patch(
                facecolor="#2ecc71",
                edgecolor="#95a5a6",
                linewidth=3,
                alpha=0.4,
                label="Prominence",
            ),
            Patch(
                facecolor="#2ecc71",
                edgecolor="#95a5a6",
                linewidth=3,
                alpha=0.6,
                label="Interval",
            ),
            Patch(
                facecolor="#2ecc71",
                edgecolor="#95a5a6",
                linewidth=3,
                alpha=0.8,
                label="Penalty",
            ),
        ]

        # Plot stacked bars for each score component
        for idx_pos, idx in enumerate(sorted_indices):
            is_best = idx == best_index
            base_color = "#2ecc71" if is_best else "#95a5a6"

            # Score components
            prom = prominence_scores[idx]
            intv = interval_scores[idx]
            pnlt = penalty_scores[idx]

            # Create stacked bars
            ax_scores.barh(
                y_pos[idx_pos], prom, bar_height, color=base_color, alpha=0.4
            )
            ax_scores.barh(
                y_pos[idx_pos], intv, bar_height, left=prom, color=base_color, alpha=0.6
            )
            ax_scores.barh(
                y_pos[idx_pos],
                pnlt,
                bar_height,
                left=prom + intv,
                color=base_color,
                alpha=0.8,
            )

            # Add total score label
            total = scores[idx]
            total_str = f"{total:.3f}" if not np.isinf(total) else "∞"
            ax_scores.text(
                prom + intv + pnlt + 0.005,
                y_pos[idx_pos],
                total_str,
                va="center",
                fontsize=9,
                fontweight="bold" if is_best else "normal",
                color="#2ecc71" if is_best else "#34495e",
            )

        # Y-axis labels showing highcut frequencies
        ax_scores.set_yticks(y_pos)
        freq_labels = []
        for i in sorted_indices:
            is_best = i == best_index
            label = f"{self.signal_processing.signal_highcut_fss[i]} Hz"
            if is_best:
                label = f"★ {label}"
            freq_labels.append(label)
        ax_scores.set_yticklabels(freq_labels, fontweight="bold")

        # Update y-axis label colors
        for idx_pos, idx in enumerate(sorted_indices):
            is_best = idx == best_index
            if is_best:
                ax_scores.get_yticklabels()[idx_pos].set_color("#2ecc71")

        ax_scores.set_xlabel(
            "Score Components (Lower is Better)", fontsize=10, fontweight="bold"
        )
        ax_scores.set_title("Score Breakdown by Highcut Frequency", fontsize=11, pad=8)

        # Add legend
        legend = ax_scores.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=9,
            framealpha=0.9,
            title="Green=Best, Gray=Others",
        )
        legend.get_title().set_fontsize(8)

        ax_scores.grid(True, axis="x", alpha=0.2, linestyle="--")
        ax_scores.spines["top"].set_visible(False)
        ax_scores.spines["right"].set_visible(False)

        # Main title
        fig.suptitle(
            "Processed Signal Selection - Best Filter Configuration Analysis",
            fontsize=15,
            fontweight="bold",
            y=0.995,
        )

        plt.tight_layout()
        plt.show()
