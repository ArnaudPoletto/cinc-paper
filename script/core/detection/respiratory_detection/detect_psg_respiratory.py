"""
This script detects respiratory events from polysomnography respiratory signals.

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
from cinc.core.detection.respiratory_detector import RespiratoryDetector
from cinc.data.data_paths import get_db2_parts_participant_processed_file_paths
from cinc.config import DATA_PATH


def _detect_psg_respiratory(
    participant_file_path: str,
    signal_data: np.ndarray,
    fs: float,
    channel_idx: int,
) -> None:
    """
    Detect respiratory events from a single polysomnography respiratory channel using
    adaptive ensemble detection.

    Args:
        participant_file_path (str): Path to the participant's processed data file.
        signal_data (np.ndarray): The respiratory signal data for a single channel.
        fs (float): The sampling frequency of the signal.
        channel_idx (int): The index of the current channel being processed.
    """
    # Check if file already exists
    results_file_path = participant_file_path.replace(
        "processed", "respiratory_detection"
    ).replace(".h5", f"_psg_respiratory_{channel_idx}.h5")
    if os.path.exists(results_file_path):
        print(
            f"✅ Respiratory detection results already exist for {participant_file_path} channel {channel_idx}. Skipping processing."
        )
        return

    # Load respiratory detector configuration
    respiratory_detector_config_file_path = (
        f"{DATA_PATH}/respiratory_detection/config/psg.yaml"
    )
    respiratory_detector = RespiratoryDetector(
        config_file_path=respiratory_detector_config_file_path,
        detect_both_phases=True,
    )

    # DB2P: Short recordings can be processed directly
    results = respiratory_detector.run(
        signal_data=signal_data,
        fs=fs,
    )

    # Write results to HDF5 file
    os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
    with h5py.File(results_file_path, "w") as f:
        _dict_to_h5py_group(results, f)


def main() -> None:
    """
    Main execution function for polysomnography respiratory detection.
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

        # Extract polysomnography respiratory signal and metadata
        signal_data = participant_data["psg"]["respiratory"]["processed"]["signal"]
        fs = participant_data["psg"]["respiratory"]["processed"]["fs"]
        n_channels = signal_data.shape[0]

        # Process each channel independently
        for channel_idx in range(n_channels):
            channel_signal_data = signal_data[channel_idx]

            _detect_psg_respiratory(
                participant_file_path=participant_file_path,
                signal_data=channel_signal_data,
                fs=fs,
                channel_idx=channel_idx,
            )


if __name__ == "__main__":
    main()
