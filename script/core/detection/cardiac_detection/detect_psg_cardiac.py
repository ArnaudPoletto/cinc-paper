"""
This script detects cardiac events from polysomnography cardiac signals.

It processes polysomnography data from DB2 Parts and saves the detection
results to HDF5 files.

Authors: Arnaud Poletto
"""

import os
import h5py
import numpy as np
from tqdm import tqdm

from cinc.utils.h5 import _dict_to_h5py_group
from cinc.data.db import get_participant_data
from cinc.core.detection.cardiac_detector import CardiacDetector
from cinc.data.data_paths import get_db2_parts_participant_processed_file_paths
from cinc.config import DATA_PATH


def _detect_psg_cardiac(
    participant_file_path: str,
    signal_data: np.ndarray,
    upsampled_signal_data: np.ndarray,
    fs: float,
    upsampled_fs: float,
) -> None:
    """
    Detect cardiac events from polysomnography cardiac signals using template matching.

    Args:
        participant_file_path (str): Path to the participant's processed data file.
        signal_data (np.ndarray): The cardiac signal data.
        upsampled_signal_data (np.ndarray): The upsampled cardiac signal data for template matching.
        fs (float): The sampling frequency of the signal.
        upsampled_fs (float): The upsampled sampling frequency.
    """
    # Check if file already exists
    results_file_path = participant_file_path.replace(
        "processed", "cardiac_detection"
    ).replace(".h5", "_psg_cardiac.h5")
    if os.path.exists(results_file_path):
        print(
            f"✅ Cardiac detection results already exist for {participant_file_path}. Skipping processing."
        )
        return

    # Load cardiac detector configuration
    cardiac_detector_config_file_path = (
        f"{DATA_PATH}/cardiac_detection/config/psg.yaml"
    )
    cardiac_detector = CardiacDetector(
        config_file_path=cardiac_detector_config_file_path,
    )

    # DB2P: Short recordings can be processed directly
    results = cardiac_detector.run(
        signal_data=signal_data,
        upsampled_signal_data=upsampled_signal_data,
        fs=fs,
        upsampled_fs=upsampled_fs,
    )

    # Write results to HDF5 file
    os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
    with h5py.File(results_file_path, "w") as f:
        _dict_to_h5py_group(results, f)


def main() -> None:
    """
    Main execution function for polysomnography cardiac detection.
    """
    participant_processed_file_paths = get_db2_parts_participant_processed_file_paths()

    # Process each participant
    for participant_file_path in tqdm(
        participant_processed_file_paths,
        desc="⌛ Processing participants...",
        position=0,
    ):
        # Load participant data
        participant_data = get_participant_data(participant_file_path)

        # Extract polysomnography cardiac signal and metadata
        signal_data = participant_data["psg"]["cardiac"]["processed"]["signal"][0]
        fs = participant_data["psg"]["cardiac"]["processed"]["fs"]
        if "upsampled" in participant_data["psg"]["cardiac"]:
            upsampled_signal_data = participant_data["psg"]["cardiac"]["upsampled"][
                "signal"
            ][0]
            upsampled_fs = participant_data["psg"]["cardiac"]["upsampled"]["fs"]
        else:
            upsampled_signal_data = None
            upsampled_fs = None

        _detect_psg_cardiac(
            participant_file_path=participant_file_path,
            signal_data=signal_data,
            upsampled_signal_data=upsampled_signal_data,
            fs=fs,
            upsampled_fs=upsampled_fs,
        )


if __name__ == "__main__":
    main()
