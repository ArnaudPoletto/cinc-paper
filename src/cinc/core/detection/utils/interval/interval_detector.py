import numpy as np
from typing import Dict


class IntervalDetector:
    def __init__(
        self,
        config: Dict,
        cardiac_mode: bool,
        debug_print: bool = False,
    ) -> None:
        if "min_interval_s" not in config:
            raise ValueError(
                "❌ 'min_interval_s' must be specified in the config dictionary."
            )
        if "coarse_window_size" not in config:
            raise ValueError(
                "❌ 'coarse_window_size' must be specified in the config dictionary."
            )
        if "coarse_tolerance" not in config:
            raise ValueError(
                "❌ 'coarse_tolerance' must be specified in the config dictionary."
            )
        if cardiac_mode:
            if "fine_tolerance" not in config:
                raise ValueError(
                    "❌ 'fine_tolerance' must be specified in the config dictionary for cardiac mode."
                )
            if "fine_window_sizes" not in config:
                raise ValueError(
                    "❌ 'fine_window_sizes' must be specified in the config dictionary for cardiac mode."
                )
            if "min_valid_ratio" not in config:
                raise ValueError(
                    "❌ 'min_valid_ratio' must be specified in the config dictionary for cardiac mode."
                )

        super(IntervalDetector, self).__init__()

        self.config = config
        self.cardiac_mode = cardiac_mode
        self.debug_print = debug_print

        self.min_interval_s = config["min_interval_s"]
        self.max_interval_s = config["max_interval_s"]
        self.coarse_window_size = config["coarse_window_size"]
        self.coarse_tolerance = config["coarse_tolerance"]
        if cardiac_mode:
            self.fine_tolerance = config["fine_tolerance"]
            self.fine_window_sizes = config["fine_window_sizes"]
            self.min_valid_ratio = config["min_valid_ratio"]

    def _coarse_filter(
        self,
        durations_s: np.ndarray,
        validity_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Coarse local filtering with fixed window.

        For each interval, compute local median using a fixed window size.
        Mark as invalid if deviation > coarse_tolerance * local_median.

        Args:
            durations_s (np.ndarray): Array of interval durations in seconds.
            validity_mask (np.ndarray): Current validity mask of intervals.

        Returns:
            new_validity_mask (np.ndarray): Updated validity mask after coarse filtering.
        """
        n_intervals = len(durations_s)
        half_window = self.coarse_window_size // 2
        new_validity_mask = validity_mask.copy()
        for i in range(n_intervals):
            if not validity_mask[i]:
                continue

            # Extract local window$
            start_idx = max(0, i - half_window)
            end_idx = min(n_intervals, i + half_window + 1)
            window_mask = validity_mask[start_idx:end_idx]
            local_durations = durations_s[start_idx:end_idx][window_mask]
            if len(local_durations) == 0:
                new_validity_mask[i] = False
                continue

            # Compute local median and check deviation
            local_median = np.median(local_durations)
            threshold = self.coarse_tolerance * local_median
            deviation = abs(durations_s[i] - local_median)
            if deviation > threshold:
                new_validity_mask[i] = False

        return new_validity_mask

    def _fine_filter(
        self,
        durations_s: np.ndarray,
        validity_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Pass 2: Fine adaptive filtering with expanding windows.

        For each interval:
        1. Start with smallest window size (e.g., 5)
        2. If < min_valid_ratio valid intervals in window, expand to next size
        3. Keep expanding until min_valid_ratio is met or max window reached
        4. If can't meet min_valid_ratio, mark as invalid
        5. Otherwise, check if deviation > fine_tolerance * local_median

        Args:
            durations_s (np.ndarray): Array of interval durations in seconds.
            validity_mask (np.ndarray): Current validity mask of intervals.

        Returns:
            new_validity_mask (np.ndarray): Updated validity mask after fine filtering.
        """
        n_intervals = len(durations_s)
        new_validity_mask = validity_mask.copy()
        for i in range(n_intervals):
            if not validity_mask[i]:
                continue

            # Try each window size until we have enough valid intervals
            local_durations = None
            window_valid = False
            for window_size in self.fine_window_sizes:
                half_window = window_size // 2
                start_idx = max(0, i - half_window)
                end_idx = min(n_intervals, i + half_window + 1)

                # Get durations of valid intervals in window
                window_mask = new_validity_mask[start_idx:end_idx]
                local_durations = durations_s[start_idx:end_idx][window_mask]

                # Check if we have enough valid intervals
                actual_window_size = end_idx - start_idx
                n_valid = len(local_durations)
                valid_ratio = (
                    n_valid / actual_window_size if actual_window_size > 0 else 0
                )
                if valid_ratio >= self.min_valid_ratio and n_valid >= 3:
                    window_valid = True
                    break

            # If we couldn't get enough valid intervals, mark as invalid
            if not window_valid or local_durations is None or len(local_durations) == 0:
                new_validity_mask[i] = False
                continue

            # Compute local median and check deviation
            local_median = np.median(local_durations)
            threshold = self.fine_tolerance * local_median
            deviation = abs(durations_s[i] - local_median)
            if deviation > threshold:
                new_validity_mask[i] = False

        return new_validity_mask

    def run(
        self,
        results: Dict,
        fs: float,
    ) -> Dict:
        if "detections" not in results:
            raise ValueError(
                "❌ 'detections' must be present in the results dictionary."
            )

        detections = results["detections"]
        if len(detections) < 2:
            results["intervals"] = np.empty((0, 2), dtype=int)
            return results

        # Build all consecutive interval pairs
        all_intervals = np.column_stack(
            (np.arange(len(detections) - 1), np.arange(1, len(detections)))
        )
        all_intervals_samples = np.column_stack((detections[:-1], detections[1:]))
        all_interval_durations_s = np.diff(all_intervals_samples, axis=1).flatten() / fs

        # Pass 1: Filter by absolute physiological bounds
        validity_mask = (all_interval_durations_s >= self.min_interval_s) & (
            all_interval_durations_s <= self.max_interval_s
        )

        if self.debug_print:
            n_intervals = len(all_intervals)
            n_invalid_bounds = n_intervals - np.sum(validity_mask)
            print(
                f"Pass 1 (physiological bounds): {n_invalid_bounds} intervals rejected."
            )

        # Pass 2: Coarse local filtering with fixed window
        validity_mask = self._coarse_filter(
            durations_s=all_interval_durations_s,
            validity_mask=validity_mask,
        )

        if self.debug_print:
            n_invalid_after_coarse = np.sum(~n_invalid_bounds) - np.sum(validity_mask)
            print(f"Pass 2 (coarse): {n_invalid_after_coarse} intervals rejected.")

        if self.cardiac_mode:
            # Pass 3: Fine adaptive filtering with expanding windows
            validity_mask = self._fine_filter(all_interval_durations_s, validity_mask)

            if self.debug_print:
                n_invalid_after_fine = np.sum(~n_invalid_after_coarse) - np.sum(
                    validity_mask
                )
                print(f"Pass 3 (fine): {n_invalid_after_fine} intervals rejected.")

        # Store results
        results["intervals"] = all_intervals[validity_mask]

        if self.debug_print:
            print(
                f"✅ Computed {len(results['intervals'])} valid intervals from {len(detections)} detections."
            )

        return results
