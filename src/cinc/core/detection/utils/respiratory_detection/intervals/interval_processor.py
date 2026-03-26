import numpy as np
from typing import Dict

class IntervalProcessor:
    def __init__(
        self,
    ) -> None:
        super(IntervalProcessor, self).__init__()

    def run(
        self,
        results: Dict,
    ) -> Dict:
        if results["phase_1"] is None:
            raise ValueError("❌ 'phase_1' results must be provided in the input dictionary.")
        
        p0_intervals = results["phase_0"]["intervals"]
        p0_detections = results["phase_0"]["detections"]
        p0_intervals_d = p0_detections[p0_intervals]

        p1_intervals = results["phase_1"]["intervals"]
        p1_detections = results["phase_1"]["detections"]
        p1_intervals_d = p1_detections[p1_intervals]

        intervals_d = [p0_intervals_d, p1_intervals_d]

        i = 0
        i_phase = 0
        j = 0
        j_phase = 1
        intervals_mask = [
            np.ones(p0_intervals.shape[0], dtype=bool),
            np.ones(p1_intervals.shape[0], dtype=bool),
        ]
        while i < intervals_d[i_phase].shape[0] and j < intervals_d[j_phase].shape[0]:
            # Make sure that i is always the interval with the smaller start time
            if intervals_d[i_phase][i,0] > intervals_d[j_phase][j,0]:
                i, j = j, i
                i_phase, j_phase = j_phase, i_phase
            
            # Remove first interval if they do not overlap
            if intervals_d[i_phase][i,1] < intervals_d[j_phase][j,0]:
                intervals_mask[i_phase][i] = False
                i += 1
            # Remove both intervals if the second interval is contained in the first one
            elif intervals_d[i_phase][i,1] >= intervals_d[j_phase][j,1]:
                intervals_mask[i_phase][i] = False
                intervals_mask[j_phase][j] = False
                i += 1
                j += 1
            # Keep the first interval otherwise
            else:
                i += 1

        results["phase_0"]["intervals"] = p0_intervals[intervals_mask[0]]
        results["phase_1"]["intervals"] = p1_intervals[intervals_mask[1]]

        return results
        
        