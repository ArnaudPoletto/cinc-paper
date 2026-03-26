"""
Interval estimator for detection sequences.

This module provides the IntervalEstimator class which estimates local intervals
between detections, detects gaps where intervals are unusually large, and predicts
positions of missing detections within those gaps.

Authors: Arnaud Poletto
"""

import numpy as np
from typing import Tuple


class IntervalEstimator:
    """
    A class for estimating intervals between detections and detecting gaps in detection sequences.
    """

    def __init__(self) -> None:
        """
        Initialize the interval estimator.
        """
        super(IntervalEstimator, self).__init__()

    def estimate_local_intervals(
        self,
        detections: np.ndarray,
        fs: float,
        window_s: float = 30.0,
    ) -> np.ndarray:
        """
        Estimate local intervals between detections using a sliding window approach.

        For each detection, this method calculates the median interval of all detections
        within a centered window around that detection. This provides a locally adaptive
        estimate of the expected interval that can handle gradual changes in detection rate.

        Args:
            detections (np.ndarray): The array of detection indices in samples.
            fs (float): The sampling frequency in Hz.
            window_s (float, optional): The window size in seconds for local interval estimation. Defaults to 30.0.

        Returns:
            estimated_intervals (np.ndarray): The array of estimated intervals (one per consecutive detection pair) in samples.
        """
        if len(detections) < 2:
            return np.array([])

        # Calculate all intervals between consecutive detections
        intervals_samples = np.diff(detections)
        median_intervals = np.median(intervals_samples)

        window_samples = int(window_s * fs)
        estimated_intervals = np.zeros(len(detections) - 1)
        for i in range(len(detections) - 1):
            current_detection = detections[i]

            # Find detections within window around current detection
            window_start = current_detection - window_samples // 2
            window_end = current_detection + window_samples // 2

            # Find which detections fall within this window
            in_window_mask = (detections >= window_start) & (detections <= window_end)
            window_detections = detections[in_window_mask]

            if len(window_detections) >= 2:
                window_intervals = np.diff(window_detections)
                estimated_intervals[i] = np.median(window_intervals)
            else:
                # Fallback to global median if not enough data in window
                estimated_intervals[i] = median_intervals

        return estimated_intervals

    def detect_gaps(
        self,
        detections: np.ndarray,
        estimated_intervals: np.ndarray,
        gap_multiplier: float = 1.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect gaps in detection sequences where actual intervals significantly exceed expected intervals.

        Args:
            detections (np.ndarray): The array of detection indices in samples.
            estimated_intervals (np.ndarray): The array of estimated intervals in samples (one per consecutive detection pair).
            gap_multiplier (float, optional): The multiplier threshold for gap detection. Defaults to 1.5.

        Returns:
            gap_start_detections (np.ndarray): The array of detection indices where gaps start.
            gap_multipliers (np.ndarray): The array of ratios between actual and expected intervals for each detected gap.
        """
        if len(detections) < 2 or len(estimated_intervals) == 0:
            return np.array([]), np.array([])

        # Calculate actual intervals
        actual_intervals = np.diff(detections)

        # Find where actual intervals are significantly larger than expected
        gap_ratios = actual_intervals / estimated_intervals
        gap_mask = gap_ratios >= gap_multiplier

        # Get the detections where gaps start and their multipliers
        gap_start_detections = detections[:-1][gap_mask]
        gap_multipliers = gap_ratios[gap_mask]

        return gap_start_detections, gap_multipliers

    def predict_missing_detections(
        self,
        gap_start_detection: int,
        gap_end_detection: int,
        gap_multiplier: float,
        max_fill_count: int = 2,
    ) -> np.ndarray:
        """
        Predict the positions of missing detections within a gap by evenly dividing the gap.

        Args:
            gap_start_detection (int): The detection index where the gap starts (in samples).
            gap_end_detection (int): The detection index where the gap ends (in samples).
            gap_multiplier (float): The ratio between the actual gap size and expected interval.
            max_fill_count (int, optional): The maximum number of detections to fill in a single gap. Defaults to 2.

        Returns:
            predicted_detections (np.ndarray): The array of predicted detection indices (in samples) within the gap.
        """
        # Determine how many detections are likely missing
        n_missing = int(np.round(gap_multiplier)) - 1
        if n_missing <= 0:
            return np.array([], dtype=int)

        # If estimated missing detections exceed max_fill_count, skip filling entirely
        if n_missing > max_fill_count:
            return np.array([], dtype=int)

        # Calculate predicted detections by dividing the gap evenly
        gap_size = gap_end_detection - gap_start_detection
        predicted_detections = []
        for i in range(1, n_missing + 1):
            # Position based on equal division of the gap
            predicted_detection = gap_start_detection + (i * gap_size) // (
                n_missing + 1
            )
            predicted_detections.append(predicted_detection)

        return np.array(predicted_detections, dtype=int)
