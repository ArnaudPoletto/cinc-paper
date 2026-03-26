"""
Amplitude-corrected cross-correlation for cardiac detection.

This module provides a function to compute cross-correlation between two signals
with amplitude correction based on quantile ranges. This helps reduce false
matches when signal amplitudes differ significantly.

Authors: Arnaud Poletto
"""

import numpy as np
from typing import Tuple


def amplitude_corrected_correlate(
    signal1: np.ndarray,
    signal2: np.ndarray,
    penalty_range: Tuple[float, float],
    mode: str = "full",
    quantile_percent: float = 5.0,
) -> np.ndarray:
    """
    Compute amplitude-corrected cross-correlation between two signals.

    This function calculates the cross-correlation between two signals and applies
    an amplitude correction penalty based on the ratio of their quantile ranges.
    When signal amplitudes differ significantly, the correlation is penalized to
    reduce false positive matches.

    Args:
        signal1 (np.ndarray): The first signal array.
        signal2 (np.ndarray): The second signal array.
        penalty_range (Tuple[float, float]): The range of penalty multipliers (min_penalty, max_penalty).
        mode (str, optional): The correlation mode ('full', 'valid', or 'same'). Defaults to "full".
        quantile_percent (float, optional): The quantile percentage for range calculation. Defaults to 5.0.

    Returns:
        corrected_correlation (np.ndarray): The amplitude-corrected correlation array.
    """
    if len(signal1) == 0 or len(signal2) == 0:
        return np.array([])

    # Compute standard cross-correlation
    correlation = np.correlate(signal1, signal2, mode=mode)

    # Only return correlation if no correction
    if penalty_range[0] == 1.0 and penalty_range[1] == 1.0:
        return correlation

    # Calculate amplitude penalty based on quantile ranges
    low_quantile = quantile_percent
    high_quantile = 100 - quantile_percent
    range1 = np.percentile(signal1, high_quantile) - np.percentile(signal1, low_quantile)
    range2 = np.percentile(signal2, high_quantile) - np.percentile(signal2, low_quantile)

    if range1 == 0 or range2 == 0:
        amplitude_penalty = penalty_range[1]
    else:
        ratio = min(range1, range2) / max(range1, range2)
        amplitude_penalty = penalty_range[0] + (penalty_range[1] - penalty_range[0]) * ratio

    # Apply amplitude correction to all correlation values
    corrected_correlation = correlation * amplitude_penalty

    return corrected_correlation


