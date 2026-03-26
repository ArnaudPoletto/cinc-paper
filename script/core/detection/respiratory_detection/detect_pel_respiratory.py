"""
This script detects respiratory events from piezoelectric sensor signals.

It processes piezoelectric sensor data from DB2 Parts and saves the detection
results to HDF5 files.

Authors: Arnaud Poletto
"""

import os
import h5py
import numpy as np
from tqdm import tqdm
from typing import List

from cinc.utils.h5 import _dict_to_h5py_group
from cinc.data.db import get_participant_data
from cinc.core.detection.respiratory_detector import RespiratoryDetector
from cinc.data.data_paths import get_db2_parts_participant_processed_file_paths
from cinc.config import DATA_PATH


def _detect_pel_respiratory(
    participant_file_path: str,
    signal_data_list: List[np.ndarray],
    fs: float,
) -> None:
    """
    Detect respiratory events from piezoelectric sensor signals using adaptive ensemble detection.

    Args:
        participant_file_path (str): Path to the participant's processed data file.
        signal_data_list (List[np.ndarray]): List of respiratory signal data from all sensors.
        fs (float): The sampling frequency of the signal.
    """
    # Check if all files already exist
    files_exist = True
    results_file_path = participant_file_path.replace("processed", "respiratory_detection")
    for individual_results_file_paths in [
        results_file_path.replace(".h5", f"_pel_respiratory_{i}.h5")
        for i in range(len(signal_data_list))
    ] + [results_file_path.replace(".h5", "_pel_respiratory_ensemble.h5")]:
        if not os.path.exists(individual_results_file_paths):
            files_exist = False
            break
    if files_exist:
        print(
            f"✅ Respiratory detection results already exist for {participant_file_path}. Skipping processing."
        )
        return

    # Load respiratory detector configuration
    respiratory_detector_config_file_path = (
        f"{DATA_PATH}/respiratory_detection/config/pel.yaml"
    )
    respiratory_detector = RespiratoryDetector(
        config_file_path=respiratory_detector_config_file_path,
        detect_both_phases=True,
    )

    # DB2P: Short recordings can be processed with ensemble directly
    individual_results_list, ensemble_results = respiratory_detector.run_ensemble(
        signal_data_list=signal_data_list,
        fs=fs,
    )

    # Write results to HDF5 files
    os.makedirs(os.path.dirname(results_file_path), exist_ok=True)

    # Save individual sensor results
    for i, individual_results in enumerate(individual_results_list):
        individual_results_file_path = results_file_path.replace(
            ".h5", f"_pel_respiratory_{i}.h5"
        )
        with h5py.File(individual_results_file_path, "w") as f:
            _dict_to_h5py_group(individual_results, f)

    # Save ensemble results
    ensemble_results_file_path = results_file_path.replace(
        ".h5", "_pel_respiratory_ensemble.h5"
    )
    with h5py.File(ensemble_results_file_path, "w") as f:
        _dict_to_h5py_group(ensemble_results, f)


def main() -> None:
    """
    Main execution function for piezoelectric respiratory detection.
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

        # Extract piezoelectric respiratory signal and metadata
        signal = participant_data["pel"]["respiratory"]["processed"]["signal"]
        signal_data_list = [signal[i] for i in range(signal.shape[0])]
        fs = participant_data["pel"]["respiratory"]["processed"]["fs"]
        _detect_pel_respiratory(
            participant_file_path=participant_file_path,
            signal_data_list=signal_data_list,
            fs=fs,
        )


if __name__ == "__main__":
    main()
