import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple
from pathlib import Path
from scipy.optimize import linear_sum_assignment

from cinc.data.db import get_participant_data
from cinc.data.data_paths import get_db2_parts_participant_processed_file_paths

CARDIAC_INTERVAL_MATCHING_COST_THRESHOLD = 1.0
CARDIAC_INTERVAL_MATCHING_MAX_VALID_DURATION_DIFF_S = 0.030
CARDIAC_INTERVAL_MATCHING_NO_OVERLAP_PENALTY = 1.0


def _get_useful_participant_data(participant_file_path: str) -> Dict:
    participant_data = get_participant_data(
        participant_file_path,
        with_psg_cardiac_detection=True,
        with_psg_respiratory_detection=False,
        with_pel_cardiac_detection=True,
        with_pel_respiratory_detection=False,
        with_pre_respiratory_detection=False,
        with_base_data=True,
    )

    # PEL cardiac
    pel_cardiac_sensor_intervals_list = []
    for key, value in participant_data["pel"]["cardiac"]["detection"].items():
        if not key.startswith("sensor_"):
            continue

        intervals = value.get("phase_0", {}).get(
            "intervals", np.array([]).reshape(0, 2)
        )
        detections = value.get("phase_0", {}).get("detections", np.array([]))
        intervals = detections[intervals]
        pel_cardiac_sensor_intervals_list.append(intervals)

    pel_cardiac_ensemble_intervals = (
        participant_data["pel"]["cardiac"]["detection"]["ensemble"]
        .get("phase_0", {})
        .get("intervals", np.array([]).reshape(0, 2))
    )
    pel_cardiac_ensemble_detections = (
        participant_data["pel"]["cardiac"]["detection"]["ensemble"]
        .get("phase_0", {})
        .get("detections", np.array([]))
    )
    pel_cardiac_ensemble_intervals = pel_cardiac_ensemble_detections[
        pel_cardiac_ensemble_intervals
    ]

    pel_cardiac_signal = participant_data["pel"]["cardiac"]["processed"]["signal"]

    # PSG cardiac
    psg_cardiac_intervals = (
        participant_data["psg"]["cardiac"]["detection"]["channel_0"]
        .get("phase_0", {})
        .get("intervals", np.array([]).reshape(0, 2))
    )
    psg_cardiac_detections = (
        participant_data["psg"]["cardiac"]["detection"]["channel_0"]
        .get("phase_0", {})
        .get("detections", np.array([]))
    )
    psg_cardiac_intervals = psg_cardiac_detections[psg_cardiac_intervals]

    data = {
        "psg": {
            "cardiac": {
                "intervals": psg_cardiac_intervals,
                "signal": participant_data["psg"]["cardiac"]["processed"]["signal"],
                "processed_fs": participant_data["psg"]["cardiac"]["processed"]["fs"],
                "upsampled_fs": participant_data["psg"]["cardiac"]["upsampled"]["fs"],
            },
        },
        "pel": {
            "cardiac": {
                "sensor_intervals_list": pel_cardiac_sensor_intervals_list,
                "ensemble_intervals": pel_cardiac_ensemble_intervals,
                "signal": pel_cardiac_signal,
                "signal_length": pel_cardiac_signal.shape[1],
                "processed_fs": participant_data["pel"]["cardiac"]["processed"]["fs"],
                "upsampled_fs": participant_data["pel"]["cardiac"]["upsampled"]["fs"],
            },
        },
    }

    return data


def get_structured_participant_data_df() -> pd.DataFrame:
    participant_file_paths = get_db2_parts_participant_processed_file_paths()
    print(f"✅ Found {len(participant_file_paths)} participant files.")

    structured_participant_data = []
    for participant_file_path in tqdm(
        participant_file_paths, desc="⏳ Loading participant data..."
    ):
        participant_name = Path(participant_file_path).stem.lower()
        participant_name = participant_name.replace("2mm_glue", "2mm-glue").replace(
            "2mm_no-glue", "2mm-no-glue"
        )
        participant_position = participant_name.split("_")[-1]
        participant_mattress = participant_name.split("_")[-2]
        if participant_mattress in [
            "surcouche",
            "sousmatelas",
            "forme",
            "matelas",
            "sol",
            "3mm-medical",
            "2mm-glue",
            "2mm-no-glue",
            "3mm",
            "3mm-medical",
        ]:
            participant_mattress = (
                participant_name.split("_")[-3] + "_" + participant_mattress
            )
            participant_id = participant_name.split("_")[-5]
        else:
            participant_id = participant_name.split("_")[-4]
        participant_mattress = participant_mattress
        participant_id = participant_id

        participant_data = _get_useful_participant_data(participant_file_path)

        structured_participant_data.append(
            {
                "id": participant_id,
                "mattress": participant_mattress,
                "position": participant_position,
                **participant_data,
            }
        )

    structured_participant_data_df = pd.DataFrame(structured_participant_data)
    structured_participant_data_df = pd.json_normalize(
        structured_participant_data, sep="_"
    )

    return structured_participant_data_df


class Rate:
    def __init__(self, start_index, end_index, fs):
        self.start_index = start_index
        self.end_index = end_index
        self.fs = fs

        self.start_s = self.start_index / self.fs
        self.end_s = self.end_index / self.fs

        self.duration = self.end_index - self.start_index
        self.duration_s = self.duration / self.fs


def get_interval_rate_estimation_from_intervals(
    intervals: np.ndarray,
    fs: float,
) -> list[Rate]:
    if intervals.shape[0] == 0:
        return []

    rates = [Rate(start, end, fs=fs) for start, end in intervals]

    return rates


def apply_interval_rate_estimation(df: pd.DataFrame) -> None:
    df["psg_cardiac_rates"] = df.progress_apply(
        lambda row: get_interval_rate_estimation_from_intervals(
            intervals=row["psg_cardiac_intervals"],
            fs=row["psg_cardiac_upsampled_fs"],
        ),
        axis=1,
    )

    df["pel_cardiac_sensor_rates_list"] = df.progress_apply(
        lambda row: [
            get_interval_rate_estimation_from_intervals(
                intervals=sensor_intervals,
                fs=row["pel_cardiac_upsampled_fs"],
            )
            for sensor_intervals in row["pel_cardiac_sensor_intervals_list"]
        ],
        axis=1,
    )

    df["pel_cardiac_ensemble_rates"] = df.progress_apply(
        lambda row: get_interval_rate_estimation_from_intervals(
            intervals=row["pel_cardiac_ensemble_intervals"],
            fs=row["pel_cardiac_upsampled_fs"],
        ),
        axis=1,
    )


def matching_cost(
    rate1: Rate,
    rate2: Rate,
    no_overlap_penalty: float,
):
    total_cost = 0.0

    # Temporal IoU
    intersection_start = max(rate1.start_s, rate2.start_s)
    intersection_end = min(rate1.end_s, rate2.end_s)
    intersection = max(0, intersection_end - intersection_start)

    union = rate1.duration_s + rate2.duration_s - intersection
    temporal_iou = intersection / union if union > 0 else 0
    temporal_cost = 1 - temporal_iou
    total_cost += temporal_cost

    # Penalize if they don't overlap in time
    if temporal_iou == 0:
        total_cost += no_overlap_penalty

    return total_cost


def get_interval_matching(
    rates1: List[Rate],
    rates2: List[Rate],
    cost_threshold: float,
    max_valid_duration_diff_s: float,
    no_overlap_penalty: float,
) -> Tuple[float, List[Tuple[int, int]]]:
    n1 = len(rates1)
    n2 = len(rates2)
    cost_matrix = np.zeros((n1, n2))
    for i, rate1 in enumerate(rates1):
        for j, rate2 in enumerate(rates2):
            cost_matrix[i, j] = matching_cost(
                rate1,
                rate2,
                no_overlap_penalty=no_overlap_penalty,
            )

    # Find optimal assignment using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < cost_threshold:
            matches.append(
                {"rate1": rates1[i], "rate2": rates2[j], "cost": cost_matrix[i, j]}
            )
    if len(matches) == 0:
        return [], 0.0

    # Find valid matches
    valid_matches = []
    for match in matches:
        duration_diff_s = abs(
            match["rate1"].duration_s - match["rate2"].duration_s
        )
        if duration_diff_s > max_valid_duration_diff_s:
            continue

        valid_matches.append(match)

    valid_matches_pct = len(valid_matches) / n1 * 100.0

    return valid_matches, valid_matches_pct


def apply_interval_matching(df: pd.DataFrame) -> None:
    # Cardiac sensor
    pel_cardiac_sensor_results = df.progress_apply(
        lambda row: [
            get_interval_matching(
                rates1=row["psg_cardiac_rates"],
                rates2=sensor_rates,
                cost_threshold=CARDIAC_INTERVAL_MATCHING_COST_THRESHOLD,
                max_valid_duration_diff_s=CARDIAC_INTERVAL_MATCHING_MAX_VALID_DURATION_DIFF_S,
                no_overlap_penalty=CARDIAC_INTERVAL_MATCHING_NO_OVERLAP_PENALTY,
            )
            for sensor_rates in row["pel_cardiac_sensor_rates_list"]
        ],
        axis=1,
    )
    df["pel_cardiac_sensor_interval_matches_list"] = pel_cardiac_sensor_results.apply(
        lambda x: [item[0] for item in x]
    )
    df["pel_cardiac_sensor_interval_matchings"] = pel_cardiac_sensor_results.apply(
        lambda x: [item[1] for item in x]
    )

    # Cardiac ensemble
    pel_cardiac_ensemble_result = df.progress_apply(
        lambda row: get_interval_matching(
            rates1=row["psg_cardiac_rates"],
            rates2=row["pel_cardiac_ensemble_rates"],
            cost_threshold=CARDIAC_INTERVAL_MATCHING_COST_THRESHOLD,
            max_valid_duration_diff_s=CARDIAC_INTERVAL_MATCHING_MAX_VALID_DURATION_DIFF_S,
            no_overlap_penalty=CARDIAC_INTERVAL_MATCHING_NO_OVERLAP_PENALTY,
        ),
        axis=1,
    )
    df["pel_cardiac_ensemble_interval_matches"] = pel_cardiac_ensemble_result.apply(
        lambda x: x[0]
    )
    df["pel_cardiac_ensemble_interval_matching"] = pel_cardiac_ensemble_result.apply(
        lambda x: x[1]
    )


def apply_matching_processing(df: pd.DataFrame) -> pd.DataFrame:
    print("Applying interval rate estimation...")
    apply_interval_rate_estimation(df)
    print("Applying interval matching...")
    apply_interval_matching(df)

    return df
