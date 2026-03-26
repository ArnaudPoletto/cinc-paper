"""
Utility function to sort detection results by detection sample index.

This module provides a function to ensure detections and their associated
arrays (prominences, are_filled) are sorted in ascending order by sample
index, and intervals are recomputed accordingly.

This is necessary because chunk merging and filtering operations may result
in unsorted detections, which can cause negative interval calculations
downstream.

Authors: Arnaud Poletto
"""

import numpy as np
from typing import Dict


def sort_detection_results(results: Dict) -> Dict:
    """
    Sort detection results by detection sample index in ascending order.

    This function ensures that:
    1. Detections are sorted in ascending order by sample index
    2. Prominences and are_filled arrays are reordered to match
    3. Interval indices are remapped to reference the new sorted positions

    Args:
        results (Dict): Detection results dictionary containing:
            - detections (np.ndarray): Array of detection sample indices
            - prominences (np.ndarray, optional): Array of prominence values
            - are_filled (np.ndarray, optional): Boolean array of gap-fill flags
            - intervals (np.ndarray, optional): Array of interval index pairs (N, 2)

    Returns:
        sorted_results (Dict): Detection results with all arrays sorted by detection index.

    Raises:
        ValueError: If intervals are already computed in results.
    """
    if "intervals" in results:
        raise ValueError(
            "Detection sorting should be called before intervals are computed."
        )
    
    if "detections" not in results:
        return results

    detections = results["detections"]
    if len(detections) == 0:
        return results

    # Get sort order
    sort_order = np.argsort(detections)
    if np.array_equal(sort_order, np.arange(len(detections))):
        return results

    sorted_results = {}
    sorted_results["detections"] = detections[sort_order]

    if "prominences" in results and results["prominences"] is not None:
        sorted_results["prominences"] = results["prominences"][sort_order]

    if "are_filled" in results and results["are_filled"] is not None:
        sorted_results["are_filled"] = results["are_filled"][sort_order]

    # Copy any other keys that weren't processed
    for key in results:
        if key not in sorted_results:
            sorted_results[key] = results[key]

    return sorted_results


def sort_phase_results(results: Dict) -> Dict:
    """
    Sort detection results for all phases in a results dictionary.

    This handles the standard output format with phase_0 and optionally phase_1.

    Args:
        results (Dict): Results dictionary that may contain phase_0 and phase_1 keys.

    Returns:
        sorted_results (Dict): Results with all phase detections sorted.
    """
    sorted_results = {}

    for key, value in results.items():
        if key in ["phase_0", "phase_1"] and value is not None:
            sorted_results[key] = sort_detection_results(value)
        else:
            sorted_results[key] = value

    return sorted_results
