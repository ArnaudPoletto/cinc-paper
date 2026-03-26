"""
Script to split DB2 dataset participant data into time-windowed parts.

This script takes processed DB2 participant files and splits them into multiple
time-based segments for each position (prone, right, left, supine), saving each segment as
a separate HDF5 file for subsequent analysis.

Authors: Arnaud Poletto
"""

import os
import h5py
import numpy as np
from tqdm import tqdm
from typing import Dict
from pathlib import Path

from cinc.utils.signal_processor import SignalProcessor
from cinc.utils.h5 import _dict_to_h5py_group
from cinc.data.data_paths import get_db2_participant_processed_file_paths
from cinc.data.db import get_participant_data
from cinc.config import (
    PROCESSED_DATA_PATH,
    DB2_PARTS_S,
    DB2_PART_NAMES,
)


def get_split_signal(
    signal: np.ndarray,
    fs: float,
    part_start_s: float,
    part_end_s: float,
    normalize: bool = True,
) -> np.ndarray:
    """
    Extract and optionally normalize a time segment from a multi-channel signal.

    This function extracts a temporal segment from a signal array and optionally
    applies robust normalization to each channel independently.

    Args:
        signal (np.ndarray): Input signal array with shape (n_channels, n_samples).
        fs (float): Sampling frequency in Hz.
        part_start_s (float): Start time of the segment in seconds.
        part_end_s (float): End time of the segment in seconds.
        normalize (bool): Whether to apply robust normalization. Defaults to True.

    Returns:
        np.ndarray: Extracted signal segment with shape (n_channels, segment_length).
    """
    # Calculate time indices
    start_idx = int(part_start_s * fs)
    end_idx = int(part_end_s * fs)
    end_idx = min(end_idx, signal.shape[1])
    split_length = end_idx - start_idx

    # Extract and optionally normalize each channel
    split_signal = np.zeros((signal.shape[0], split_length))
    for i in range(signal.shape[0]):
        windowed_signal = signal[i, start_idx:end_idx]
        if normalize:
            windowed_signal = SignalProcessor.normalize_signal(
                windowed_signal, method="robust"
            )
        split_signal[i, :] = windowed_signal

    return split_signal


def split_db2_participant_data(
    participant_data: Dict,
    participant_file_path: str,
) -> None:
    """
    Split a single participant's data into time-based parts and save each part.

    This function divides participant data into predefined temporal segments
    (e.g., baseline, task, recovery) and saves each segment as a separate HDF5 file.

    Args:
        participant_data (Dict): Complete participant data dictionary.
        participant_file_path (str): Path to the source participant file.
    """
    # Extract sampling frequencies
    psg_cardiac_raw_fs = participant_data["psg"]["cardiac"]["raw"]["fs"]
    psg_cardiac_processed_fs = participant_data["psg"]["cardiac"]["processed"]["fs"]
    psg_cardiac_upsampled_fs = participant_data["psg"]["cardiac"]["upsampled"]["fs"]
    psg_respiratory_raw_fs = participant_data["psg"]["respiratory"]["raw"]["fs"]
    psg_respiratory_processed_fs = participant_data["psg"]["respiratory"]["processed"][
        "fs"
    ]
    pel_raw_fs = participant_data["pel"]["raw"]["fs"]
    pel_cardiac_processed_fs = participant_data["pel"]["cardiac"]["processed"]["fs"]
    pel_cardiac_upsampled_fs = participant_data["pel"]["cardiac"]["upsampled"]["fs"]
    pel_respiratory_processed_fs = participant_data["pel"]["respiratory"]["processed"][
        "fs"
    ]
    pre_raw_fs = participant_data["pre"]["raw"]["fs"]
    pre_respiratory_processed_fs = participant_data["pre"]["respiratory"]["processed"][
        "fs"
    ]

    # Split data into parts and save each segment
    for (part_start_s, part_end_s), part_name in zip(DB2_PARTS_S, DB2_PART_NAMES):
        # Create data structure for this part
        participant_data_part = {
            "psg": {
                "cardiac": {
                    "raw": {
                        "signal": get_split_signal(
                            participant_data["psg"]["cardiac"]["raw"]["signal"],
                            psg_cardiac_raw_fs,
                            part_start_s,
                            part_end_s,
                            normalize=False,
                        ),
                        "fs": psg_cardiac_raw_fs,
                    },
                    "processed": {
                        "signal": get_split_signal(
                            participant_data["psg"]["cardiac"]["processed"]["signal"],
                            psg_cardiac_processed_fs,
                            part_start_s,
                            part_end_s,
                            normalize=False,
                        ),
                        "fs": psg_cardiac_processed_fs,
                    },
                    "upsampled": {
                        "signal": get_split_signal(
                            participant_data["psg"]["cardiac"]["upsampled"]["signal"],
                            psg_cardiac_upsampled_fs,
                            part_start_s,
                            part_end_s,
                            normalize=False,
                        ),
                        "fs": psg_cardiac_upsampled_fs,
                    },
                },
                "respiratory": {
                    "raw": {
                        "signal": get_split_signal(
                            participant_data["psg"]["respiratory"]["raw"]["signal"],
                            psg_respiratory_raw_fs,
                            part_start_s,
                            part_end_s,
                            normalize=False,
                        ),
                        "fs": psg_respiratory_raw_fs,
                    },
                    "processed": {
                        "signal": get_split_signal(
                            participant_data["psg"]["respiratory"]["processed"][
                                "signal"
                            ],
                            psg_respiratory_processed_fs,
                            part_start_s,
                            part_end_s,
                            normalize=False,
                        ),
                        "fs": psg_respiratory_processed_fs,
                    },
                },
            },
            "pel": {
                "raw": {
                    "signal": get_split_signal(
                        participant_data["pel"]["raw"]["signal"],
                        pel_raw_fs,
                        part_start_s,
                        part_end_s,
                        normalize=False,
                    ),
                    "fs": pel_raw_fs,
                },
                "cardiac": {
                    "processed": {
                        "signal": get_split_signal(
                            participant_data["pel"]["cardiac"]["processed"]["signal"],
                            pel_cardiac_processed_fs,
                            part_start_s,
                            part_end_s,
                            normalize=False,
                        ),
                        "fs": pel_cardiac_processed_fs,
                    },
                    "upsampled": {
                        "signal": get_split_signal(
                            participant_data["pel"]["cardiac"]["upsampled"]["signal"],
                            pel_cardiac_upsampled_fs,
                            part_start_s,
                            part_end_s,
                            normalize=False,
                        ),
                        "fs": pel_cardiac_upsampled_fs,
                    },
                },
                "respiratory": {
                    "processed": {
                        "signal": get_split_signal(
                            participant_data["pel"]["respiratory"]["processed"][
                                "signal"
                            ],
                            pel_respiratory_processed_fs,
                            part_start_s,
                            part_end_s,
                            normalize=False,
                        ),
                        "fs": pel_respiratory_processed_fs,
                    },
                },
            },
            "pre": {
                "raw": {
                    "signal": get_split_signal(
                        participant_data["pre"]["raw"]["signal"],
                        pre_raw_fs,
                        part_start_s,
                        part_end_s,
                        normalize=False,
                    ),
                    "fs": pre_raw_fs,
                },
                "respiratory": {
                    "processed": {
                        "signal": get_split_signal(
                            participant_data["pre"]["respiratory"]["processed"][
                                "signal"
                            ],
                            pre_respiratory_processed_fs,
                            part_start_s,
                            part_end_s,
                            normalize=False,
                        ),
                        "fs": pre_respiratory_processed_fs,
                    },
                },
            },
        }

        # Save part to HDF5 file
        participant_name = Path(participant_file_path).stem
        participant_data_part_file_path = (
            f"{PROCESSED_DATA_PATH}/db2_parts/{participant_name}_{part_name.lower()}.h5"
        )
        os.makedirs(os.path.dirname(participant_data_part_file_path), exist_ok=True)
        with h5py.File(participant_data_part_file_path, "w") as f:
            _dict_to_h5py_group(participant_data_part, f)


def main() -> None:
    """
    Execute the DB2 data splitting pipeline.

    This function processes all DB2 participant files and splits each one
    into predefined temporal segments.
    """
    participant_processed_file_paths = get_db2_participant_processed_file_paths()
    print(
        f"ℹ️  Found {len(participant_processed_file_paths)} participant processed files."
    )

    for participant_file_path in tqdm(
        participant_processed_file_paths, desc="⌛ Splitting DB2 data..."
    ):
        participant_data = get_participant_data(participant_file_path)
        split_db2_participant_data(participant_data, participant_file_path)


if __name__ == "__main__":
    main()
