"""
Missing detection filler for detection sequences.

This module provides the MissingDetectionFiller class which identifies and fills
gaps in detection sequences by estimating missing detections based on local
interval patterns and optionally snapping them to nearby peaks.

Authors: Arnaud Poletto
"""

import numpy as np
from typing import Dict, Any, Optional

from cinc.core.detection.utils.fill.interval_estimator import IntervalEstimator
from cinc.core.detection.utils.fill.filled_detection_snapper import FilledDetectionSnapper


class MissingDetectionFiller:
    """
    A class for filling missing detections in detection sequences using interval-based gap detection.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        debug_print: bool = False,
    ) -> None:
        """
        Initialize the missing detection filler with configuration parameters.

        Args:
            config (Dict[str, Any]): The configuration dictionary with required parameters.
            debug_print (bool, optional): Whether to enable debug printing. Defaults to False.

        Raises:
            ValueError: If 'interval_estimation_window_s' is missing from config.
            ValueError: If 'gap_detection_multiplier' is missing from config.
            ValueError: If 'max_fill_count' is missing from config.
        """
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

        super(MissingDetectionFiller, self).__init__()

        self.debug_print = debug_print
        self.interval_estimation_window_s = config["interval_estimation_window_s"]
        self.gap_detection_multiplier = config["gap_detection_multiplier"]
        self.max_fill_count = config["max_fill_count"]

        # Initialize interval estimator
        self.interval_estimator = IntervalEstimator()

        # Initialize filled detection snapper
        filled_detection_snapper_config = {
            "min_snap_distance_s": config["min_snap_distance_s"],
        }
        self.filled_detection_snapper = FilledDetectionSnapper(
            config=filled_detection_snapper_config
        )

    def run(
        self,
        results: Dict,
        fs: float,
        snapping_detections: Optional[np.ndarray] = None,
        snapping_prominences: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Fill missing detections in detection sequences using interval estimation.

        This method identifies gaps in the detection sequence by comparing actual
        intervals with estimated local intervals, then predicts and fills missing
        detections. Optionally snaps filled detections to nearby peaks.

        Args:
            results (Dict): The results dictionary containing detections and prominences.
            fs (float): The sampling frequency in Hz.
            snapping_detections (Optional[np.ndarray], optional): The array of peak indices for snapping. Defaults to None.
            snapping_prominences (Optional[np.ndarray], optional): The array of peak prominences for snapping. Defaults to None.

        Returns:
            updated_results (Dict): The updated results dictionary with filled detections and are_filled flag array.

        Raises:
            ValueError: If 'detections' is missing from results.
            ValueError: If 'prominences' is missing from results.
        """
        if "detections" not in results:
            raise ValueError(
                "❌ 'detections' must be present in the results dictionary."
            )
        if "prominences" not in results:
            raise ValueError(
                "❌ 'prominences' must be present in the results dictionary."
            )

        detections = results["detections"]
        prominences = results["prominences"]

        if len(detections) < 3:
            if self.debug_print:
                print(
                    f"⚠️  Too few detections ({len(detections)}) for gap filling, skipping..."
                )
            results_copy = results.copy()
            results_copy["are_filled"] = np.zeros(len(detections), dtype=bool)
            return results_copy

        # Estimate local intervals for each detection
        estimated_intervals = self.interval_estimator.estimate_local_intervals(
            detections=detections,
            fs=fs,
            window_s=self.interval_estimation_window_s,
        )

        if len(estimated_intervals) == 0:
            if self.debug_print:
                print(
                    "⚠️  No estimated intervals available for gap filling, skipping..."
                )
            results_copy = results.copy()
            results_copy["are_filled"] = np.zeros(len(detections), dtype=bool)
            return results_copy

        # Detect gaps
        gap_start_detections, gap_multipliers = self.interval_estimator.detect_gaps(
            detections=detections,
            estimated_intervals=estimated_intervals,
            gap_multiplier=self.gap_detection_multiplier,
        )

        if len(gap_start_detections) == 0:
            if self.debug_print:
                print("✅ No gaps detected, no filling needed.")
            results_copy = results.copy()
            results_copy["are_filled"] = np.zeros(len(detections), dtype=bool)
            return results_copy

        # Fill detected gaps
        filled_detections = []
        for gap_start_detection, gap_multiplier in zip(
            gap_start_detections, gap_multipliers
        ):
            # Find the corresponding gap end index
            gap_start_pos = np.where(detections == gap_start_detection)[0][0]
            if gap_start_pos + 1 >= len(detections):
                continue

            # Predict missing detections
            gap_end_detection = detections[gap_start_pos + 1]
            predicted_detections = self.interval_estimator.predict_missing_detections(
                gap_start_detection=gap_start_detection,
                gap_end_detection=gap_end_detection,
                gap_multiplier=gap_multiplier,
                max_fill_count=self.max_fill_count,
            )

            if len(predicted_detections) > 0:
                filled_detections.extend(predicted_detections)

        if len(filled_detections) == 0:
            if self.debug_print:
                print("✅ No missing detections predicted, no filling done.")
            # Add are_filled array even when no filling is done (all False)
            results_copy = results.copy()
            results_copy["are_filled"] = np.zeros(len(detections), dtype=bool)
            return results_copy

        filled_detections = np.array(filled_detections, dtype=int)

        # Combine detections and sort
        all_detections = np.concatenate([detections, filled_detections])
        all_prominences = np.concatenate(
            [prominences, np.zeros(len(filled_detections))]
        )
        sort_order = np.argsort(all_detections)
        final_detections = all_detections[sort_order]
        final_prominences = all_prominences[sort_order]

        # Create are_filled boolean array
        original_are_filled = np.zeros(len(detections), dtype=bool)
        filled_are_filled = np.ones(len(filled_detections), dtype=bool)
        all_are_filled = np.concatenate([original_are_filled, filled_are_filled])
        final_are_filled = all_are_filled[sort_order]

        # Update results (keep original correlation arrays unchanged)
        results = results.copy()
        results["detections"] = final_detections
        results["prominences"] = final_prominences
        results["are_filled"] = final_are_filled

        # Snap filled detections to nearby peaks if requested
        if snapping_detections is not None and snapping_prominences is not None:
            results = self.filled_detection_snapper.run(
                results=results,
                snapping_detections=snapping_detections,
                snapping_prominences=snapping_prominences,
                fs=fs,
            )

        if self.debug_print:
            n_filled = len(filled_detections)
            print(f"✅ Filled {n_filled} missing detections.")

        return results
