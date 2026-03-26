"""
Signal quality scoring for cardiac detection sequences.

This module provides functions to compute quality scores for processed cardiac
signals based on prominence variation, interval regularity, and detection count
reliability.

Authors: Arnaud Poletto
"""

import numpy as np
from typing import Dict


def get_processed_signal_score_dict(
    detections: np.ndarray,
    prominences: np.ndarray,
    prominence_alpha: float,
    interval_alpha: float,
    penalty_alpha: float,
    eps: float = 1e-6,
) -> Dict:
    """
    Compute a quality score for a processed cardiac signal based on detection characteristics.

    This function evaluates signal quality using three components: prominence variation
    (consistency of peak heights), interval variation (regularity of detection spacing),
    and a penalty term that accounts for the reliability of the estimate based on the
    number of detections. Lower scores indicate better quality.

    Args:
        detections (np.ndarray): The array of detection indices in samples.
        prominences (np.ndarray): The array of prominence values for each detection.
        prominence_alpha (float): The weight for prominence variation in the final score.
        interval_alpha (float): The weight for interval variation in the final score.
        penalty_alpha (float): The weight for the penalty term in the final score.
        eps (float, optional): The small constant to prevent division by zero. Defaults to 1e-6.

    Returns:
        score_dict (Dict): A dictionary containing individual score components and the final score.
    """
    if len(detections) < 2:
        return {
            "prominence_score": np.nan,
            "interval_score": np.nan,
            "penalty_score": np.nan,
            "score": np.nan,
        }

    # Prominence variation
    median_prominence = np.median(prominences)
    mad_prominence = np.median(np.abs(prominences - median_prominence))
    prominence_variation = mad_prominence / (median_prominence + eps)

    # Interval variation
    intervals = np.diff(detections)
    median_interval = np.median(intervals)
    mad_interval = np.median(np.abs(intervals - median_interval))
    interval_variation = mad_interval / (median_interval + eps)

    # Penalty for unreliable estimates
    penalty = 1.0 / np.sqrt(len(detections))
    score = (
        prominence_alpha * prominence_variation
        + interval_alpha * interval_variation
        + penalty_alpha * penalty
    )

    return {
        "prominence_score": prominence_variation,
        "interval_score": interval_variation,
        "penalty_score": penalty,
        "score": score,
    }