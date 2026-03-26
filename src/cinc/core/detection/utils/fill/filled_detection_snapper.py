"""
Filled detection snapper for refining predicted detection positions.

This module provides the FilledDetectionSnapper class which snaps filled
(predicted) detections to nearby peak positions if they are within a
specified distance threshold.

Authors: Arnaud Poletto
"""

import numpy as np
from typing import Dict


class FilledDetectionSnapper:
    """
    A class for snapping filled detections to nearby peaks to improve detection accuracy.
    """

    def __init__(self, config: Dict) -> None:
        """
        Initialize the filled detection snapper with configuration parameters.

        Args:
            config (Dict): The configuration dictionary with required parameters.

        Raises:
            ValueError: If 'min_snap_distance_s' is missing from config.
        """
        if "min_snap_distance_s" not in config:
            raise ValueError(
                "❌ 'min_snap_distance_s' must be specified in the config dictionary."
            )

        super(FilledDetectionSnapper, self).__init__()

        self.config = config
        self.min_snap_distance_s = config["min_snap_distance_s"]

    def _remove_unsnapped_detections(
        self,
        results: Dict,
        filled_to_keep: np.ndarray,
    ) -> Dict:
        """
        Remove filled detections that could not be snapped to nearby peaks.

        Args:
            results (Dict): The results dictionary containing detections, prominences, and are_filled arrays.
            filled_to_keep (np.ndarray): The array of indices indicating which filled detections to keep.

        Returns:
            updated_results (Dict): The results dictionary with unsnapped detections removed.
        """
        filled_positions = np.where(results["are_filled"])[0]
        if len(filled_to_keep) == len(filled_positions):
            return results

        kept_filled_positions = filled_positions[filled_to_keep]

        detections_to_keep = np.ones(len(results["detections"]), dtype=bool)
        detections_to_keep[filled_positions] = False  # Remove all filled
        detections_to_keep[kept_filled_positions] = True  # Add back snapped filled

        # Apply the mask to all arrays
        results["detections"] = results["detections"][detections_to_keep]
        results["prominences"] = results["prominences"][detections_to_keep]
        results["are_filled"] = results["are_filled"][detections_to_keep]

        return results

    def _remove_duplicate_detections(
        self,
        results: Dict,
    ) -> Dict:
        """
        Remove duplicate detections that may have been created during the snapping process.

        When multiple filled detections snap to the same peak, this method keeps only
        the first occurrence and removes duplicates.

        Args:
            results (Dict): The results dictionary containing detections, prominences, and are_filled arrays.

        Returns:
            updated_results (Dict): The results dictionary with duplicates removed.
        """
        unique_detections, unique_idx_map = np.unique(
            results["detections"], return_index=True
        )
        if len(unique_detections) < len(results["detections"]):
            results["detections"] = unique_detections
            results["prominences"] = results["prominences"][unique_idx_map]
            results["are_filled"] = results["are_filled"][unique_idx_map]

        return results

    def run(
        self,
        results: Dict,
        snapping_detections: np.ndarray,
        snapping_prominences: np.ndarray,
        fs: float,
    ) -> Dict:
        """
        Snap filled detections to nearby peaks within the distance threshold.

        This method finds the nearest peak for each filled detection and snaps it to
        that peak if the distance is within the configured threshold. Filled detections
        that cannot be snapped are removed, and any duplicates created by snapping are
        removed.

        Args:
            results (Dict): The results dictionary containing detections, prominences, and are_filled arrays.
            snapping_detections (np.ndarray): The array of peak indices to snap to.
            snapping_prominences (np.ndarray): The array of peak prominences corresponding to snapping_detections.
            fs (float): The sampling frequency in Hz.

        Returns:
            updated_results (Dict): The updated results dictionary with snapped detections.

        Raises:
            ValueError: If 'detections' is missing from results.
            ValueError: If 'prominences' is missing from results.
            ValueError: If 'are_filled' is missing from results.
        """
        if "detections" not in results:
            raise ValueError(
                "❌ 'detections' must be present in the results dictionary."
            )
        if "prominences" not in results:
            raise ValueError(
                "❌ 'prominences' must be present in the results dictionary."
            )
        if "are_filled" not in results:
            raise ValueError(
                "❌ 'are_filled' must be present in the results dictionary."
            )

        are_filled = results["are_filled"]
        filled_detections = results["detections"][are_filled]

        if len(snapping_detections) == 0 or len(filled_detections) == 0:
            return results, []

        # Snap filled detections to nearest detection within threshold
        snapped_filled_detections = []
        snapped_filled_prominences = []
        filled_to_keep = []
        for i, detection in enumerate(filled_detections):
            distances = np.abs(snapping_detections - detection)
            nearest_idx = np.argmin(distances)
            nearest_detection = snapping_detections[nearest_idx]
            nearest_distance = np.abs(nearest_detection - detection)
            if nearest_distance > int(self.min_snap_distance_s * fs):
                continue

            snapped_filled_detections.append(nearest_detection)
            snapped_filled_prominences.append(snapping_prominences[nearest_idx])
            filled_to_keep.append(i)

        # Remove filled detections that didn't snap
        results = self._remove_unsnapped_detections(
            results=results,
            filled_to_keep=np.array(filled_to_keep, dtype=int),
        )

        # Update the snapped filled detections with their new detections
        if len(snapped_filled_detections) > 0:
            are_filled = results["are_filled"]
            results["detections"][are_filled] = np.array(
                snapped_filled_detections, dtype=int
            )
            results["prominences"][are_filled] = np.array(
                snapped_filled_prominences, dtype=float
            )

        # Remove duplicates that may have been created during snapping
        results = self._remove_duplicate_detections(results=results)

        return results
