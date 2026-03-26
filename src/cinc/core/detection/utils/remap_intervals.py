import numpy as np


def remap_intervals(
    intervals: np.ndarray,
    indices_mask: np.ndarray,
) -> np.ndarray:
    """
    Remap intervals after filtering detections.

    Intervals that reference removed detections (where indices_mask is False)
    are dropped. Remaining interval indices are remapped to match the new
    positions in the filtered detections array.

    Args:
        intervals (np.ndarray): Array of shape (N, 2) containing pairs of indices
            into the original detections array.
        indices_mask (np.ndarray): Boolean mask of shape (M,) indicating which
            detections were kept (True) or removed (False).

    Returns:
        new_indices (np.ndarray): Array of shape (K, 2) with remapped interval indices, where
            K <= N. Intervals referencing any removed detection are excluded.
    """
    if len(intervals) == 0:
        return np.array([], dtype=int).reshape(0, 2)

    # Keep only intervals where both endpoints were kept
    valid = indices_mask[intervals[:, 0]] & indices_mask[intervals[:, 1]]
    intervals = intervals[valid]

    if len(intervals) == 0:
        return np.array([], dtype=int).reshape(0, 2)

    # Build mapping from old indices to new indices (cumsum of mask)
    new_indices = np.cumsum(indices_mask) - 1

    return new_indices[intervals]