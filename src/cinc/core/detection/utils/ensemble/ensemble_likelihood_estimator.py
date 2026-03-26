"""
Ensemble likelihood estimator for combining detection results from multiple signals.

This module provides the EnsembleLikelihoodEstimator class which computes a
combined likelihood signal from multiple individual signal detection results
using weighted Gaussian summation and optional phase reordering.

Authors: Arnaud Poletto
"""

import copy
import numpy as np
from scipy.stats import norm
from sklearn.cluster import DBSCAN
from typing import List, Dict, Tuple, Optional

from cinc.core.detection.utils.remap_intervals import remap_intervals


class EnsembleLikelihoodEstimator:
    """A class for estimating ensemble likelihood from multiple signal detection results."""

    def __init__(
        self,
        config: Dict,
        detect_both_phases: bool = False,
        debug_print: bool = False,
    ) -> None:
        """
        Initialize the ensemble likelihood estimator with configuration parameters.

        Args:
            config (Dict): The configuration dictionary with required parameters.
            detect_both_phases (bool, optional): Whether to detect both phases. Defaults to False.
            debug_print (bool, optional): Whether to enable debug printing. Defaults to False.

        Raises:
            ValueError: If 'min_tolerance_s' is missing from config.
            ValueError: If 'max_tolerance_interval_ratio' is missing from config.
            ValueError: If 'mad_threshold' is missing from config.
            ValueError: If 'weight_temperature' is missing from config.
            ValueError: If 'default_sigma_s' is missing from config.
            ValueError: If 'sigma_scaling_factor' is missing from config.
            ValueError: If 'min_sigma_s' is missing from config.
            ValueError: If 'max_sigma_s' is missing from config.
            ValueError: If 'processed_signal_alpha' is missing from config.
            ValueError: If 'prominence_alpha' is missing from config.
            ValueError: If 'synchronize_detections' is missing from config.
        """
        if "min_tolerance_s" not in config:
            raise ValueError(
                "❌ 'min_tolerance_s' must be specified in the config dictionary."
            )
        if "max_tolerance_interval_ratio" not in config:
            raise ValueError(
                "❌ 'max_tolerance_interval_ratio' must be specified in the config dictionary."
            )
        if "mad_threshold" not in config:
            raise ValueError(
                "❌ 'mad_threshold' must be specified in the config dictionary."
            )
        if "weight_temperature" not in config:
            raise ValueError(
                "❌ 'weight_temperature' must be specified in the config dictionary."
            )
        if "default_sigma_s" not in config:
            raise ValueError(
                "❌ 'default_sigma_s' must be specified in the config dictionary."
            )
        if "sigma_scaling_factor" not in config:
            raise ValueError(
                "❌ 'sigma_scaling_factor' must be specified in the config dictionary."
            )
        if "min_sigma_s" not in config:
            raise ValueError(
                "❌ 'min_sigma_s' must be specified in the config dictionary."
            )
        if "max_sigma_s" not in config:
            raise ValueError(
                "❌ 'max_sigma_s' must be specified in the config dictionary."
            )
        if "processed_signal_alpha" not in config:
            raise ValueError(
                "❌ 'processed_signal_alpha' must be specified in the config dictionary."
            )
        if "prominence_alpha" not in config:
            raise ValueError(
                "❌ 'prominence_alpha' must be specified in the config dictionary."
            )
        if "synchronize_detections" not in config:
            raise ValueError(
                "❌ 'synchronize_detections' must be specified in the config dictionary."
            )

        super(EnsembleLikelihoodEstimator, self).__init__()

        self.detect_both_phases = detect_both_phases
        self.debug_print = debug_print
        self.min_tolerance_s = config["min_tolerance_s"]
        self.max_tolerance_interval_ratio = config["max_tolerance_interval_ratio"]
        self.mad_threshold = config["mad_threshold"]
        self.weight_temperature = config["weight_temperature"]
        self.default_sigma_s = config["default_sigma_s"]
        self.sigma_scaling_factor = config["sigma_scaling_factor"]
        self.min_sigma_s = config["min_sigma_s"]
        self.max_sigma_s = config["max_sigma_s"]
        self.processed_signal_alpha = config["processed_signal_alpha"]
        self.prominence_alpha = config["prominence_alpha"]
        self.synchronize_detections = config["synchronize_detections"]

    def _get_global_median_interval_length_s(
        self,
        results_list: List[Dict],
        fs: float,
    ) -> Optional[float]:
        """
        Compute the global median interval length across all signals.

        Args:
            results_list (List[Dict]): The list of detection results from each signal.
            fs (float): The sampling frequency in Hz.

        Returns:
            global_median_interval_length_s (Optional[float]): The global median interval
            length in seconds, or None if insufficient intervals found.
        """
        all_intervals = []
        for results in results_list:
            if self.detect_both_phases:
                phase_0_intervals = results["phase_0"]["intervals"]
                phase_0_detections = results["phase_0"]["detections"]
                phase_0_intervals = phase_0_detections[phase_0_intervals]

                phase_1_intervals = results["phase_1"]["intervals"]
                phase_1_detections = results["phase_1"]["detections"]
                phase_1_intervals = phase_1_detections[phase_1_intervals]

                intervals = np.concatenate([phase_0_intervals, phase_1_intervals])
            else:
                intervals = results["phase_0"]["intervals"]
                detections = results["phase_0"]["detections"]
                intervals = detections[intervals]

            if len(intervals) > 0:
                all_intervals.extend(intervals.tolist())

        if len(all_intervals) < 2:
            if self.debug_print:
                print(
                    "⚠️  Not enough intervals found across signals to compute global median interval length."
                )
            return None

        all_intervals = np.array(all_intervals)
        all_interval_lengths_s = np.diff(all_intervals) / fs
        global_median_interval_length_s = np.median(all_interval_lengths_s)

        if self.debug_print:
            print(
                f"✅ Computed global median interval length: {global_median_interval_length_s:.3f} seconds."
            )

        return global_median_interval_length_s

    def _get_signal_median_interval_lengths_s(
        self,
        results_list: List[Dict],
        fs: float,
    ) -> List[Optional[float]]:
        """
        Compute the median interval length for each signal.

        Args:
            results_list (List[Dict]): The list of detection results from each signal.
            fs (float): The sampling frequency in Hz.

        Returns:
            median_interval_lengths_s (List[Optional[float]]): The list of median interval
            lengths in seconds for each signal, or None for signals with insufficient intervals.
        """
        median_interval_lengths_s = []
        for results in results_list:
            if self.detect_both_phases:
                phase_0_intervals = results["phase_0"]["intervals"]
                phase_0_detections = results["phase_0"]["detections"]
                phase_0_intervals = phase_0_detections[phase_0_intervals]

                phase_1_intervals = results["phase_1"]["intervals"]
                phase_1_detections = results["phase_1"]["detections"]
                phase_1_intervals = phase_1_detections[phase_1_intervals]

                intervals = np.concatenate([phase_0_intervals, phase_1_intervals])
            else:
                intervals = results["phase_0"]["intervals"]
                detections = results["phase_0"]["detections"]
                intervals = detections[intervals]

            if len(intervals) >= 2:
                interval_lengths_s = np.diff(intervals) / fs
                median_interval_length_s = np.median(interval_lengths_s)
                median_interval_lengths_s.append(median_interval_length_s)
            else:
                median_interval_lengths_s.append(None)

        if self.debug_print:
            print("✅ Computed signal median interval lengths:")
            for i, median_length in enumerate(median_interval_lengths_s):
                if median_length is None:
                    print(
                        f"\tSignal {i}: No intervals found to compute median interval length."
                    )
                else:
                    print(f"\tSignal {i}: {median_length:.3f} seconds.")

        return median_interval_lengths_s

    def _get_valid_signal_indices(
        self,
        global_median_interval_length_s: float,
        median_interval_lengths_s: List[Optional[float]],
        max_tolerance_s: float,
    ) -> List[int]:
        """
        Identify valid signals based on interval length consistency.

        This method filters signals whose median interval length is within
        a tolerance of the global median interval length.

        Args:
            global_median_interval_length_s (float): The global median interval length in seconds.
            median_interval_lengths_s (List[Optional[float]]): The list of median interval
            lengths in seconds for each signal.
            max_tolerance_s (float): The maximum tolerance in seconds.

        Returns:
            valid_signal_indices (List[int]): The list of indices of valid signals.
        """
        # Get tolerance based on MAD of median interval lengths
        valid_medians = [m for m in median_interval_lengths_s if m is not None]
        if len(valid_medians) == 0:
            if self.debug_print:
                print("⚠️  No valid median interval lengths found to compute tolerance.")
            return []
        elif len(valid_medians) == 1:
            if self.debug_print:
                print(
                    "⚠️  Only one valid median interval length found; using max tolerance."
                )
            tolerance_s = max_tolerance_s
        else:
            mad = np.median(
                np.abs(np.array(valid_medians) - global_median_interval_length_s)
            )
            tolerance_s = np.clip(
                mad * self.mad_threshold,
                self.min_tolerance_s,
                max_tolerance_s,
            )
            if self.debug_print:
                print(
                    f"➡️ Using tolerance for valid signals: {tolerance_s:.3f} seconds."
                )

        # Determine valid signals
        valid_signal_indices = [
            signal_idx
            for signal_idx, median_length in enumerate(median_interval_lengths_s)
            if median_length is not None
            and abs(median_length - global_median_interval_length_s) <= tolerance_s
        ]

        if self.debug_print:
            print(
                f"✅ Found {len(valid_signal_indices)}/{len(median_interval_lengths_s)} valid signal detections: {valid_signal_indices}"
            )

        return valid_signal_indices

    def _get_optimal_shift(
        self,
        reference_detections: np.ndarray,
        signal_detections: np.ndarray,
        fs: float,
        max_shift_s: float,
    ) -> int:
        """
        Find the optimal shift to align signal detections with reference detections.

        This method searches for the shift that minimizes the average distance
        from shifted signal detections to their nearest reference detections.

        Args:
            reference_detections (np.ndarray): The array of reference detection indices.
            signal_detections (np.ndarray): The array of signal detection indices to shift.
            fs (float): The sampling frequency in Hz.
            max_shift_s (float): The maximum shift in seconds.

        Returns:
            best_shift (int): The optimal shift in samples.
        """
        max_shift = int(max_shift_s * fs)
        ref_sorted = np.sort(reference_detections)
        best_shift = 0
        min_average_distance = np.inf
        for shift in np.linspace(
            -max_shift, max_shift, num=2 * max_shift + 1, dtype=int
        ):
            shifted = signal_detections + shift

            # Binary search for nearest reference detection
            idx = np.searchsorted(ref_sorted, shifted)
            idx = np.clip(idx, 1, len(ref_sorted) - 1)
            dist_left = np.abs(shifted - ref_sorted[idx - 1])
            dist_right = np.abs(shifted - ref_sorted[idx])
            average_distance = np.mean(np.minimum(dist_left, dist_right))

            # Update best shift if current is better
            if average_distance < min_average_distance:
                min_average_distance = average_distance
                best_shift = shift

        return best_shift
    
    def _get_shifted_detections(
        self,
        detections: np.ndarray,
        shift: int,
        signal_length: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply a shift to detections and return valid shifted detections.
        
        Args:
            detections (np.ndarray): The array of detection indices.
            shift (int): The shift in samples to apply.
            signal_length (int): The length of the signal.
            
        Returns:
            shifted_detections (np.ndarray): The array of valid shifted detection indices.
            mask (np.ndarray): The boolean mask indicating valid shifted detections.
        """
        shifted = detections + shift
        mask = (shifted >= 0) & (shifted < signal_length)
                                 
        return shifted[mask], mask

    def _get_synchronized_results_list(
        self,
        results_list: List[Dict],
        valid_signal_indices: List[int],
        fs: float,
        max_shift_s: float,
        signal_length: int,
    ) -> List[Dict]:
        """
        Synchronize detection results by applying optimal shifts to align with a reference.

        This method finds the signal with the most detections as the reference,
        computes optimal shifts for all other signals, and applies those shifts
        to align detections across signals.

        Args:
            results_list (List[Dict]): The list of detection results from each signal.
            valid_signal_indices (List[int]): The list of indices of valid signals.
            fs (float): The sampling frequency in Hz.
            max_shift_s (float): The maximum shift in seconds.
            signal_length (int): The length of the signal.

        Returns:
            synchronized_results_list (List[Dict]): The list of synchronized detection results.

        Raises:
            ValueError: If maximum shift is not positive.
        """
        if max_shift_s <= 0:
            raise ValueError("❌ Maximum shift must be positive.")

        processed_signal_scores = [
            results["processed_signal_score"] for results in results_list
        ]
        reference_idx = np.argmin(processed_signal_scores)
        reference_results = results_list[reference_idx]

        # Find optimal shifts for signals
        optimal_shifts = []
        for results in results_list:
            # Combine detections if they are lists (positive and negative detections)
            reference_detections = reference_results["phase_0"]["detections"]
            if self.detect_both_phases:
                reference_detections = np.concatenate(
                    [
                        reference_results["phase_0"]["detections"],
                        reference_results["phase_1"]["detections"],
                    ]
                )
            signal_detections = results["phase_0"]["detections"]
            if self.detect_both_phases:
                signal_detections = np.concatenate(
                    [results["phase_0"]["detections"], results["phase_1"]["detections"]]
                )

            optimal_shift = self._get_optimal_shift(
                reference_detections=reference_detections,
                signal_detections=signal_detections,
                fs=fs,
                max_shift_s=max_shift_s,
            )
            optimal_shifts.append(optimal_shift)

        if self.debug_print:
            print("✅ Computed optimal shifts for signals:")
            for i, shift in zip(valid_signal_indices, optimal_shifts):
                print(f"\tSignal {i}: {(shift / fs):.3f} seconds.")

        # Apply shifts to signals
        synchronized_results_list = []
        for results, shift in zip(results_list, optimal_shifts):
            shifted_detections, mask = self._get_shifted_detections(
                results["phase_0"]["detections"], shift, signal_length
            )
            # Combine detections if they are lists (positive and negative detections)
            synchronized_results = {
                "phase_0": {
                    "detections": shifted_detections,
                    "prominences": results["phase_0"]["prominences"][mask],
                    "are_filled": results["phase_0"]["are_filled"][mask],
                    "intervals": remap_intervals(results["phase_0"]["intervals"], mask),
                },
                **{k: v for k, v in results.items() if k not in ["phase_0", "phase_1"]},
            }
            if self.detect_both_phases:
                shifted_detections, mask = self._get_shifted_detections(
                    results["phase_1"]["detections"], shift, signal_length
                )
                synchronized_results["phase_1"] = {
                    "detections": shifted_detections,
                    "prominences": results["phase_1"]["prominences"][mask],
                    "are_filled": results["phase_1"]["are_filled"][mask],
                    "intervals": remap_intervals(results["phase_1"]["intervals"], mask),
                }
            else:
                synchronized_results["phase_1"] = None
            synchronized_results_list.append(synchronized_results)

        return synchronized_results_list

    def _get_weights(
        self,
        results_list: List[Dict],
        valid_signal_indices: List[int],
    ) -> List[np.ndarray]:
        """
        Compute weights for each signal based on signal quality scores.

        This method computes weights using a softmax over combined processed
        signal scores and prominence scores. If no scores are available,
        uniform weights are returned.

        Args:
            results_list (List[Dict]): The list of detection results from each signal.
            valid_signal_indices (List[int]): The list of indices of valid signals.

        Returns:
            weights (np.ndarray): The array of weights for each signal.
        """
        if "processed_signal_score" in results_list[0]:
            processed_signal_scores = []
            prominence_scores = []
            for results in results_list:
                # Processed signal score
                processed_signal_score = results["processed_signal_score"]
                processed_signal_scores.append(processed_signal_score)

                # Amplitude score
                if self.detect_both_phases:
                    amplitudes = np.concatenate(
                        [
                            results["phase_0"]["prominences"],
                            results["phase_1"]["prominences"],
                        ]
                    )
                else:
                    amplitudes = results["phase_0"]["prominences"]
                prominence_score = np.median(amplitudes) if len(amplitudes) > 0 else 0.0
                prominence_scores.append(prominence_score)

            # Normalize scores
            processed_signal_scores = np.array(processed_signal_scores, dtype=float)
            prominence_scores = np.array(prominence_scores, dtype=float)
            if np.max(processed_signal_scores) > 0:
                processed_signal_scores = (
                    1.0 - processed_signal_scores
                )  # Make higher scores better
                processed_signal_scores /= np.max(processed_signal_scores)
            if np.max(prominence_scores) > 0:
                prominence_scores /= np.max(prominence_scores)

            # Combine scores
            scores = (
                self.processed_signal_alpha * processed_signal_scores
                + self.prominence_alpha * prominence_scores
            )
            scores = scores / self.weight_temperature

            # Compute weights using softmax
            exp_scores = np.exp(scores - np.max(scores))
            weights = exp_scores / np.sum(exp_scores)
        else:
            if self.debug_print:
                print(
                    "⚠️  'processed_signal_score' not found in results, using uniform weights."
                )
            weights = np.ones(len(results_list)) / len(results_list)

        if self.debug_print:
            print("✅ Computed weights based on signal scores:")
            for i, (idx, weight) in enumerate(zip(valid_signal_indices, weights)):
                print(
                    f"\tSignal {idx}: {weight:.3f}, from scores - Processed Signal: {processed_signal_scores[i]:.3f}, Prominence: {prominence_scores[i]:.3f}"
                )

        return weights

    def _reorder_detections_by_phase_clustering(
        self,
        results_list: List[Dict],
        fs: float,
        global_median_interval_length_s: float,
    ) -> Tuple[List[Dict], List[bool]]:
        """
        Reorder detections by clustering to ensure consistent phase assignment.

        This method uses a greedy clustering approach to determine if phase_0
        and phase_1 detections should be swapped for each signal to maintain
        consistent phase alignment across all signals.

        Args:
            results_list (List[Dict]): The list of detection results from each signal.
            fs (float): The sampling frequency in Hz.
            global_median_interval_length_s (float): The global median interval length in seconds.

        Returns:
            reordered (List[Dict]): The list of reordered detection results.
            should_swap (List[bool]): The list of boolean flags indicating if phases were swapped.
        """
        # Get cluster assignment order based on number of detections
        num_detections = [
            len(results["phase_0"]["detections"])
            + len(results["phase_1"]["detections"])
            for results in results_list
        ]
        order = np.argsort(-np.array(num_detections))

        # Intitialize clusters with best sensor
        best_idx = order[0]
        cluster_0 = results_list[best_idx]["phase_0"]["detections"] / fs
        cluster_1 = results_list[best_idx]["phase_1"]["detections"] / fs

        # Track which sensors need phase swap
        should_swap = [False] * len(results_list)
        distance_threshold = global_median_interval_length_s / 2

        # Assign remaining sensors to clusters
        for sensor_idx in order[1:]:
            phase_0 = results_list[sensor_idx]["phase_0"]["detections"] / fs
            phase_1 = results_list[sensor_idx]["phase_1"]["detections"] / fs

            # Compute mean minimum distance from phase_0 to each cluster
            def mean_min_distance(detections, cluster, threshold):
                if len(detections) == 0 or len(cluster) == 0:
                    return float("inf")

                min_dists = [np.min(np.abs(cluster - det)) for det in detections]
                valid_dists = [d for d in min_dists if d < threshold]

                return np.mean(valid_dists) if valid_dists else float("inf")

            # Get cost for keeping current phase assignment and for swapping
            cost_keep = mean_min_distance(
                phase_0, cluster_0, distance_threshold
            ) + mean_min_distance(phase_1, cluster_1, distance_threshold)
            cost_swap = mean_min_distance(
                phase_0, cluster_1, distance_threshold
            ) + mean_min_distance(phase_1, cluster_0, distance_threshold)

            # Assign to cluster with lower cost
            if cost_swap < cost_keep:
                should_swap[sensor_idx] = True
                cluster_0 = np.concatenate([cluster_0, phase_1])
                cluster_1 = np.concatenate([cluster_1, phase_0])
            else:
                cluster_0 = np.concatenate([cluster_0, phase_0])
                cluster_1 = np.concatenate([cluster_1, phase_1])

        # Apply swaps
        reordered = []
        for i, res in enumerate(results_list):
            if should_swap[i]:
                r = copy.deepcopy(res)
                for k in ["detections", "prominences", "are_filled", "intervals"]:
                    r["phase_0"][k], r["phase_1"][k] = r["phase_1"][k], r["phase_0"][k]
                reordered.append(r)
            else:
                reordered.append(res)

        if self.debug_print:
            n_swapped = sum(should_swap)
            print(
                f"✅ Reordered detections by phase clustering, swapped {n_swapped}/{len(results_list)} sensors."
            )

        return reordered, should_swap

    def _get_sigma_s(
        self,
        results_list: List[Dict],
        global_median_interval_length_s: float,
        fs: float,
    ) -> float:
        """
        Compute the sigma parameter for Gaussian likelihood estimation.

        This method uses DBSCAN clustering to estimate the spread of detection
        clusters and returns a sigma value based on the median cluster spread.

        Args:
            results_list (List[Dict]): The list of detection results from each signal.
            global_median_interval_length_s (float): The global median interval length in seconds.
            fs (float): The sampling frequency in Hz.

        Returns:
            sigma_s (float): The sigma value in seconds for Gaussian likelihood estimation.
        """
        # Collect all detections
        all_detections = []
        for results in results_list:
            all_detections.extend(results["detections"])
        if len(all_detections) < 3:
            if self.debug_print:
                print(
                    "⚠️  Not enough detections found to estimate sigma, using default sigma."
                )
            return self.default_sigma_s

        # Convert to seconds
        all_detections = np.array(all_detections) / fs
        X = all_detections.reshape(-1, 1)

        eps = global_median_interval_length_s * 0.25
        dbscan = DBSCAN(eps=eps, min_samples=2)
        labels = dbscan.fit_predict(X)

        if self.debug_print:
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            print(
                f"➡️ DBSCAN found {n_clusters} clusters and {n_noise} noise points for sigma estimation."
            )

        # Compute cluster spreads
        spreads = []
        for label in set(labels):
            if label == -1:
                continue
            cluster = all_detections[labels == label]
            if len(cluster) >= 2:
                spreads.append(np.std(cluster))
        if len(spreads) == 0:
            if self.debug_print:
                print("⚠️  No clusters found for sigma estimation, using default sigma.")
            return self.default_sigma_s

        # Return median spread
        sigma_s = np.median(spreads) * self.sigma_scaling_factor
        sigma_s = np.clip(sigma_s, self.min_sigma_s, self.max_sigma_s)

        if self.debug_print:
            print(
                f"✅ Auto-estimated sigma: {sigma_s:.3f}s from {len(spreads)} clusters."
            )

        return sigma_s

    def _get_likelihood(
        self,
        results_list: List[Dict],
        weights: List[float],
        signal_length: int,
        sigma: float,
    ) -> np.ndarray:
        """
        Compute the likelihood signal by summing weighted Gaussian distributions.

        This method places a Gaussian distribution at each detection location,
        weighted by the signal weight, and sums them to produce the final
        likelihood signal.

        Args:
            results_list (List[Dict]): The list of detection results from each signal.
            weights (np.ndarray): The array of weights for each signal.
            signal_length (int): The length of the output likelihood signal.
            sigma (float): The standard deviation of Gaussian distributions in samples.

        Returns:
            likelihood (np.ndarray): The normalized likelihood signal.
        """
        # Sum Gaussian distributions for each signal's detected detections
        likelihood = np.zeros(signal_length, dtype=float)
        for results, weight in zip(results_list, weights):
            detections = results["detections"]

            if len(detections) == 0:
                continue

            for idx in detections:
                window_size = int(4 * sigma)
                start = max(0, idx - window_size)
                end = min(signal_length, idx + window_size + 1)
                sample_detections = np.arange(start, end)
                gaussian_values = norm.pdf(
                    sample_detections,
                    loc=idx,
                    scale=sigma,
                )
                likelihood[start:end] += weight * gaussian_values

        # Normalize likelihood
        if np.max(likelihood) > 0:
            likelihood /= np.max(likelihood)

        return likelihood

    def run(
        self,
        results_list: List[Dict],
        signal_length: int,
        fs: float,
        force_synchronization: bool = False,
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[float],
        Optional[np.ndarray],
    ]:
        """
        Run ensemble likelihood estimation on multiple signal detection results.

        This method computes a combined likelihood signal from individual signal
        detection results using weighted Gaussian summation. It filters invalid
        signals, optionally synchronizes detections, computes signal weights,
        and generates the final likelihood estimate.

        Args:
            results_list (List[Dict]): The list of detection results from each signal.
            signal_length (int): The length of the output likelihood signal.
            fs (float): The sampling frequency in Hz.
            force_synchronization (bool, optional): Whether to force synchronization of 
            detections. Defaults to False.

        Returns:
            likelihood (Optional[np.ndarray]): The likelihood signal or list of likelihood
            signals for dual-phase detection, or None if estimation failed.
            global_median_interval_length_s (Optional[float]): The global median interval
            length in seconds, or None if estimation failed.
            phase_0_sign (Optional[np.ndarray]): The array of phase signs for each signal,
            or None if estimation failed.

        Raises:
            ValueError: If results list is empty.
            ValueError: If signal length is not positive.
            ValueError: If sampling frequency is not positive.
        """
        if len(results_list) == 0:
            raise ValueError("❌ Results list cannot be empty.")

        if signal_length <= 0:
            raise ValueError("❌ Signal length must be positive.")

        if fs <= 0:
            raise ValueError("❌ Sampling frequency must be positive.")

        # Compute global median interval length across all signals
        global_median_interval_length_s = self._get_global_median_interval_length_s(
            results_list=results_list,
            fs=fs,
        )
        if global_median_interval_length_s is None:
            if self.debug_print:
                print(
                    "⚠️  Cannot compute ensemble likelihood without global median interval length."
                )
            return None, None, None

        # Compute median interval length per signal
        signal_median_interval_lengths_s = self._get_signal_median_interval_lengths_s(
            results_list=results_list,
            fs=fs,
        )

        # Get valid signals based on interval lengths
        max_tolerance_s = (
            global_median_interval_length_s * self.max_tolerance_interval_ratio
        )
        valid_signal_indices = self._get_valid_signal_indices(
            global_median_interval_length_s=global_median_interval_length_s,
            median_interval_lengths_s=signal_median_interval_lengths_s,
            max_tolerance_s=max_tolerance_s,
        )
        if len(valid_signal_indices) == 0:
            if self.debug_print:
                print("⚠️  No valid signals found based on interval length consistency.")
            return None, None, None
        valid_results_list = [results_list[i] for i in valid_signal_indices]

        # Get synchronized signal detections
        if self.synchronize_detections or force_synchronization:
            max_shift_s = global_median_interval_length_s / 2
            valid_results_list = self._get_synchronized_results_list(
                results_list=valid_results_list,
                valid_signal_indices=valid_signal_indices,
                fs=fs,
                max_shift_s=max_shift_s,
                signal_length=signal_length,
            )

        # Compute signal weights to combine detections
        valid_weights = self._get_weights(
            results_list=valid_results_list,
            valid_signal_indices=valid_signal_indices,
        )

        # Reorder results to match close detections
        phase_0_sign = np.zeros(len(results_list), dtype=int)
        if self.detect_both_phases:
            valid_results_list, should_swap = (
                self._reorder_detections_by_phase_clustering(
                    results_list=valid_results_list,
                    fs=fs,
                    global_median_interval_length_s=global_median_interval_length_s,
                )
            )
            # If no swapping happened, phase 0 detects maxima, else minima
            phase_0_sign[valid_signal_indices] = np.array(
                [-1 if swap else 1 for swap in should_swap]
            )

        if self.detect_both_phases:
            phase_0_results_list = [
                results["phase_0"] for results in valid_results_list
            ]
            phase_1_results_list = [
                results["phase_1"] for results in valid_results_list
            ]

            # Get final likelihood estimates for each phase
            phase_0_sigma_s = self._get_sigma_s(
                results_list=phase_0_results_list,
                global_median_interval_length_s=global_median_interval_length_s,
                fs=fs,
            )
            phase_0_sigma = phase_0_sigma_s * fs
            phase_0_likelihood = self._get_likelihood(
                results_list=phase_0_results_list,
                weights=valid_weights,
                signal_length=signal_length,
                sigma=phase_0_sigma,
            )

            phase_1_sigma_s = self._get_sigma_s(
                results_list=phase_1_results_list,
                global_median_interval_length_s=global_median_interval_length_s,
                fs=fs,
            )
            phase_1_sigma = phase_1_sigma_s * fs
            phase_1_likelihood = self._get_likelihood(
                results_list=phase_1_results_list,
                weights=valid_weights,
                signal_length=signal_length,
                sigma=phase_1_sigma,
            )

            if self.debug_print:
                print(
                    "✅ Computed final ensemble likelihood estimates for both phases."
                )

            return (
                [phase_0_likelihood, phase_1_likelihood],
                global_median_interval_length_s,
                phase_0_sign,
            )
        else:
            phase_0_results_list = [
                results["phase_0"] for results in valid_results_list
            ]

            # Get final likelihood estimate
            sigma_s = self._get_sigma_s(
                results_list=phase_0_results_list,
                global_median_interval_length_s=global_median_interval_length_s,
                fs=fs,
            )
            sigma = sigma_s * fs
            likelihood = self._get_likelihood(
                results_list=phase_0_results_list,
                weights=valid_weights,
                signal_length=signal_length,
                sigma=sigma,
            )

            if self.debug_print:
                print("✅ Computed final ensemble likelihood estimate.")

            return likelihood, global_median_interval_length_s, phase_0_sign
