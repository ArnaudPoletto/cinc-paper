"""
This script preprocesses raw DB2 participant data and converts it to HDF5 format.

It extracts and processes cardiac and respiratory signals from both piezoelectric
sensors and polysomnography data from TDMS files, and saves the processed data
to HDF5 files.

Authors: Arnaud Poletto
"""

import os
import h5py
import numpy as np
from tqdm import tqdm
from typing import Dict
from nptdms import TdmsFile

from cinc.utils.h5 import _dict_to_h5py_group
from cinc.utils.signal_processor import SignalProcessor
from cinc.data.data_paths import get_db2_participant_raw_folder_paths
from cinc.config import (
    PROCESSING_VARIABLES,
    DB2_N_PEL_SENSORS,
    DB2_N_PRE_SENSORS,
    DB2_PRE_SENSOR_NAMES,
)


# ============================================================================
# Data Extraction Functions
# ============================================================================


def _extract_pel_data(
    participant_raw_pel_data: TdmsFile,
    participant_processed_data: Dict,
) -> Dict:
    """
    Extract and process piezoelectric sensor data from TDMS file.

    This function processes raw piezoelectric signals for both cardiac
    and respiratory analysis, including filtering, resampling, and normalization.

    Args:
        participant_raw_pel_data (TdmsFile): Raw piezoelectric TDMS data.
        participant_processed_data (Dict): Processed data dictionary to update.

    Returns:
        Dict: Updated processed data dictionary with piezoelectric signals.
    """
    # Extract sampling frequency
    pel_group = participant_raw_pel_data["Untitled"]
    wf_increment = pel_group["Elec_0 (Collected)"].properties["wf_increment"]
    pel_raw_fs = 1.0 / wf_increment

    # Get processing parameters
    params = PROCESSING_VARIABLES["pel"]["cardiac"]
    lowcut_cardiac = params["lowcut"]
    highcut_cardiac = params["highcut"] if params["highcut"] is not None else np.inf

    # Process each sensor
    pel_raw_signals = []
    pel_cardiac_processed_signals = []
    pel_cardiac_upsampled_signals = []
    pel_respiratory_processed_signals = []
    for i in range(DB2_N_PEL_SENSORS):
        pel_raw_signal = pel_group[f"Elec_{i} (Collected)"][:]

        # Process cardiac signal
        cardiac = SignalProcessor.bandpass_filter(
            signal_data=pel_raw_signal,
            lowcut=lowcut_cardiac,
            highcut=highcut_cardiac,
            fs=pel_raw_fs,
            order=3,
        )

        # Resample cardiac signal to processed frequency
        pel_cardiac_processed_signal = SignalProcessor.resample_signal(
            signal_data=cardiac,
            original_fs=pel_raw_fs,
            target_fs=participant_processed_data["pel"]["cardiac"]["processed"]["fs"],
        )

        # Resample cardiac signal to upsampled frequency
        pel_cardiac_upsampled_signal = SignalProcessor.resample_signal(
            signal_data=cardiac,
            original_fs=pel_raw_fs,
            target_fs=participant_processed_data["pel"]["cardiac"]["upsampled"]["fs"],
        )

        # Process respiratory signal
        pel_respiratory_processed_signal = SignalProcessor.resample_signal(
            signal_data=pel_raw_signal,
            original_fs=pel_raw_fs,
            target_fs=participant_processed_data["pel"]["respiratory"]["processed"][
                "fs"
            ],
        )

        # Append processed signals
        pel_raw_signals.append(pel_raw_signal)
        pel_cardiac_processed_signals.append(pel_cardiac_processed_signal)
        pel_cardiac_upsampled_signals.append(pel_cardiac_upsampled_signal)
        pel_respiratory_processed_signals.append(pel_respiratory_processed_signal)

    # Stack all sensor signals
    pel_raw_signals = np.stack(pel_raw_signals, axis=0)
    pel_cardiac_processed_signals = np.stack(pel_cardiac_processed_signals, axis=0)
    pel_cardiac_upsampled_signals = np.stack(pel_cardiac_upsampled_signals, axis=0)
    pel_respiratory_processed_signals = np.stack(
        pel_respiratory_processed_signals, axis=0
    )

    # Store cardiac and respiratory data
    participant_processed_data["pel"]["raw"]["signal"] = pel_raw_signals
    participant_processed_data["pel"]["raw"]["fs"] = pel_raw_fs
    participant_processed_data["pel"]["cardiac"]["processed"]["signal"] = (
        pel_cardiac_processed_signals
    )
    participant_processed_data["pel"]["cardiac"]["upsampled"]["signal"] = (
        pel_cardiac_upsampled_signals
    )

    participant_processed_data["pel"]["respiratory"]["processed"]["signal"] = (
        pel_respiratory_processed_signals
    )
    return participant_processed_data


def _extract_pre_data(
    participant_raw_pel_data: TdmsFile,
    participant_processed_data: Dict,
) -> Dict:
    """
    Extract and process piezoresistive sensor data from TDMS file.

    This function processes raw piezoresistive signals for respiratory analysis,
    including filtering, resampling, and normalization.

    Args:
        participant_raw_pel_data (TdmsFile): Raw piezoresistive TDMS data.
        participant_processed_data (Dict): Processed data dictionary to update.

    Returns:
        Dict: Updated processed data dictionary with piezoresistive signals.
    """
    # Extract sampling frequency
    pre_group = participant_raw_pel_data["Untitled"]
    wf_increment = pre_group["Res_1LC (Collected)"].properties["wf_increment"]
    pre_raw_fs = 1.0 / wf_increment

    # Process each sensor
    pre_raw_signals = []
    pre_respiratory_processed_signals = []
    for i in range(DB2_N_PRE_SENSORS):
        pre_raw_signal = pre_group[f"{DB2_PRE_SENSOR_NAMES[i]} (Collected)"][:]

        # Process respiratory signal
        pre_respiratory_processed_signal = SignalProcessor.resample_signal(
            signal_data=pre_raw_signal,
            original_fs=pre_raw_fs,
            target_fs=participant_processed_data["pre"]["respiratory"]["processed"][
                "fs"
            ],
        )

        # Append processed signals
        pre_raw_signals.append(pre_raw_signal)
        pre_respiratory_processed_signals.append(pre_respiratory_processed_signal)

    # Stack all sensor signals
    pre_raw_signals = np.stack(pre_raw_signals, axis=0)
    pre_respiratory_processed_signals = np.stack(
        pre_respiratory_processed_signals, axis=0
    )

    # Store cardiac and respiratory data
    participant_processed_data["pre"]["raw"]["signal"] = pre_raw_signals
    participant_processed_data["pre"]["raw"]["fs"] = pre_raw_fs
    participant_processed_data["pre"]["respiratory"]["processed"]["signal"] = (
        pre_respiratory_processed_signals
    )

    return participant_processed_data


def _extract_psg_data(
    participant_raw_psg_cardiac_data: TdmsFile,
    participant_raw_psg_respiratory_data: TdmsFile,
    participant_processed_data: Dict,
) -> Dict:
    """
    Extract and process polysomnography data from TDMS files.

    This function processes raw polysomnography signals including ECG (cardiac) and
    respiratory belts from separate TDMS files.

    Args:
        participant_raw_psg_cardiac_data (TdmsFile): Raw polysomnography cardiac TDMS data.
        participant_raw_psg_respiratory_data (TdmsFile): Raw polysomnography respiratory TDMS data.
        participant_processed_data (Dict): Processed data dictionary to update.

    Returns:
        Dict: Updated processed data dictionary with polysomnography signals.

    Raises:
        ValueError: If ECG channel is not found in polysomnography data.
        FileNotFoundError: If expected respiratory channels are not found.
    """
    # Get processing parameters
    params = PROCESSING_VARIABLES["psg"]["cardiac"]
    lowcut_cardiac = params["lowcut"]
    highcut_cardiac = params["highcut"] if params["highcut"] is not None else np.inf

    # Extract cardiac data
    psg_cardiac_group = participant_raw_psg_cardiac_data["Untitled"]
    wf_increment = None
    cardiac = None
    channel_names = [channel.name for channel in psg_cardiac_group.channels()]
    if "Collected" in channel_names:
        wf_increment = psg_cardiac_group["Collected"].properties["wf_increment"]
        cardiac = psg_cardiac_group["Collected"][:]
    elif "ECG1 (Collected)" in channel_names:
        wf_increment = psg_cardiac_group["ECG1 (Collected)"].properties["wf_increment"]
        cardiac = psg_cardiac_group["ECG1 (Collected)"][:]
    else:
        raise ValueError("❌ ECG channel not found in polysomnography data.")
    psg_cardiac_raw_fs = 1.0 / wf_increment

    # Process cardiac signal
    cardiac = SignalProcessor.powerline_filter(
        signal_data=cardiac,
        fs=psg_cardiac_raw_fs,
    )
    psg_cardiac_raw_signal = cardiac.copy()

    cardiac = SignalProcessor.bandpass_filter(
        signal_data=cardiac,
        lowcut=lowcut_cardiac,
        highcut=highcut_cardiac,
        fs=psg_cardiac_raw_fs,
        order=3,
    )

    # Resample cardiac signal to processed frequency
    psg_cardiac_processed_signal = SignalProcessor.resample_signal(
        signal_data=cardiac,
        original_fs=psg_cardiac_raw_fs,
        target_fs=participant_processed_data["psg"]["cardiac"]["processed"]["fs"],
    )

    # Resample cardiac signal to upsampled frequency
    psg_cardiac_upsampled_signal = SignalProcessor.resample_signal(
        signal_data=cardiac,
        original_fs=psg_cardiac_raw_fs,
        target_fs=participant_processed_data["psg"]["cardiac"]["upsampled"]["fs"],
    )

    # Store cardiac data
    participant_processed_data["psg"]["cardiac"]["raw"]["signal"] = (
        psg_cardiac_raw_signal[np.newaxis, :]
    )
    participant_processed_data["psg"]["cardiac"]["raw"]["fs"] = psg_cardiac_raw_fs
    participant_processed_data["psg"]["cardiac"]["processed"]["signal"] = (
        psg_cardiac_processed_signal[np.newaxis, :]
    )
    participant_processed_data["psg"]["cardiac"]["upsampled"]["signal"] = (
        psg_cardiac_upsampled_signal[np.newaxis, :]
    )

    # Extract respiratory data
    psg_respiratory_group = participant_raw_psg_respiratory_data["Untitled"]
    respiratory_1 = None
    respiratory_2 = None
    channel_names = [channel.name for channel in psg_respiratory_group.channels()]
    if "Collected" in channel_names:
        wf_increment = psg_respiratory_group["Collected"].properties["wf_increment"]
        respiratory_1 = psg_respiratory_group["Collected"][:]
        respiratory_2 = psg_respiratory_group["Collected 1"][:]
    if "Resp1 (Collected)" in channel_names:
        wf_increment = psg_respiratory_group["Resp1 (Collected)"].properties[
            "wf_increment"
        ]
        respiratory_1 = psg_respiratory_group["Resp1 (Collected)"][:]
        respiratory_2 = psg_respiratory_group["Resp2 (Collected)"][:]
    psg_respiratory_raw_fs = 1.0 / wf_increment

    # Resample respiratory signals
    psg_respiratory_1_processed_signal = SignalProcessor.resample_signal(
        signal_data=respiratory_1,
        original_fs=psg_respiratory_raw_fs,
        target_fs=participant_processed_data["psg"]["respiratory"]["processed"]["fs"],
    )
    psg_respiratory_2_processed_signal = SignalProcessor.resample_signal(
        signal_data=respiratory_2,
        original_fs=psg_respiratory_raw_fs,
        target_fs=participant_processed_data["psg"]["respiratory"]["processed"]["fs"],
    )

    # Stack respiratory signals
    psg_respiratory_raw_signal = np.stack([respiratory_1, respiratory_2], axis=0)
    psg_respiratory_processed_signal = np.stack(
        [psg_respiratory_1_processed_signal, psg_respiratory_2_processed_signal], axis=0
    )

    # Store respiratory data
    participant_processed_data["psg"]["respiratory"]["raw"]["signal"] = (
        psg_respiratory_raw_signal
    )
    participant_processed_data["psg"]["respiratory"]["raw"]["fs"] = (
        psg_respiratory_raw_fs
    )
    participant_processed_data["psg"]["respiratory"]["processed"]["signal"] = (
        psg_respiratory_processed_signal
    )

    return participant_processed_data


# ============================================================================
# Main Processing Functions
# ============================================================================


def _process_participant_raw_folder(participant_raw_folder_path: str) -> None:
    """
    Process a single participant's raw data folder.

    This function loads raw TDMS data files (piezoelectric, ECG, respiratory),
    extracts and processes the signals, and saves to an HDF5 file.

    Args:
        participant_raw_folder_path (str): Path to the raw data folder.

    Raises:
        FileNotFoundError: If required TDMS files are not found in the folder.
    """
    # Check if the processed file already exists
    processed_file_path = (
        participant_raw_folder_path.replace("raw", "processed") + ".h5"
    )
    if os.path.exists(processed_file_path):
        print(
            f"ℹ️  Processed file already exists: {processed_file_path}. Skipping processing."
        )
        return

    # Extract processing parameters
    psg_cardiac_processed_fs = float(
        PROCESSING_VARIABLES["psg"]["cardiac"]["processed_fs"]
    )
    psg_cardiac_upsampled_fs = float(
        PROCESSING_VARIABLES["psg"]["cardiac"]["upsampled_fs"]
    )
    psg_respiratory_processed_fs = float(
        PROCESSING_VARIABLES["psg"]["respiratory"]["processed_fs"]
    )
    pel_cardiac_processed_fs = float(
        PROCESSING_VARIABLES["pel"]["cardiac"]["processed_fs"]
    )
    pel_cardiac_upsampled_fs = float(
        PROCESSING_VARIABLES["pel"]["cardiac"]["upsampled_fs"]
    )
    pel_respiratory_processed_fs = float(
        PROCESSING_VARIABLES["pel"]["respiratory"]["processed_fs"]
    )
    pre_respiratory_processed_fs = float(
        PROCESSING_VARIABLES["pre"]["respiratory"]["processed_fs"]
    )

    # Initialize processed data structure
    participant_processed_data = {
        "psg": {
            "cardiac": {
                "raw": {
                    "signal": None,
                    "fs": None,
                },
                "processed": {
                    "signal": None,
                    "fs": psg_cardiac_processed_fs,
                },
                "upsampled": {
                    "signal": None,
                    "fs": psg_cardiac_upsampled_fs,
                },
            },
            "respiratory": {
                "raw": {
                    "signal": None,
                    "fs": None,
                },
                "processed": {
                    "signal": None,
                    "fs": psg_respiratory_processed_fs,
                },
            },
        },
        "pel": {
            "raw": {
                "signal": None,
                "fs": None,
            },
            "cardiac": {
                "processed": {
                    "signal": None,
                    "fs": pel_cardiac_processed_fs,
                },
                "upsampled": {
                    "signal": None,
                    "fs": pel_cardiac_upsampled_fs,
                },
            },
            "respiratory": {
                "processed": {
                    "signal": None,
                    "fs": pel_respiratory_processed_fs,
                },
            },
        },
        "pre": {
            "raw": {
                "signal": None,
                "fs": None,
            },
            "respiratory": {
                "processed": {
                    "signal": None,
                    "fs": pre_respiratory_processed_fs,
                },
            },
        },
    }

    # Find and load piezoelectric data
    pel_files = [
        f for f in os.listdir(participant_raw_folder_path) if f.endswith("_piezo.tdms")
    ]
    if not pel_files:
        raise FileNotFoundError(
            f"❌ Piezo file not found in {participant_raw_folder_path}."
        )
    participant_raw_pel_file_path = os.path.join(
        participant_raw_folder_path, pel_files[0]
    )
    with TdmsFile.open(participant_raw_pel_file_path) as participant_raw_pel_data:
        participant_processed_data = _extract_pel_data(
            participant_raw_pel_data=participant_raw_pel_data,
            participant_processed_data=participant_processed_data,
        )
        participant_processed_data = _extract_pre_data(
            participant_raw_pel_data=participant_raw_pel_data,
            participant_processed_data=participant_processed_data,
        )

    # Find and load polysomnography data
    ecg_files = [
        f for f in os.listdir(participant_raw_folder_path) if f.endswith("_ECG.tdms")
    ]
    resp_files = [
        f for f in os.listdir(participant_raw_folder_path) if f.endswith("_Resp.tdms")
    ]
    if not ecg_files:
        raise FileNotFoundError(
            f"❌ ECG file not found in {participant_raw_folder_path}."
        )
    if not resp_files:
        raise FileNotFoundError(
            f"❌ RESP file not found in {participant_raw_folder_path}."
        )
    participant_raw_cardiac_file_path = os.path.join(
        participant_raw_folder_path, ecg_files[0]
    )
    participant_raw_respiratory_file_path = os.path.join(
        participant_raw_folder_path, resp_files[0]
    )
    with (
        TdmsFile.open(
            participant_raw_cardiac_file_path
        ) as participant_raw_psg_cardiac_data,
        TdmsFile.open(
            participant_raw_respiratory_file_path
        ) as participant_raw_psg_respiratory_data,
    ):
        participant_processed_data = _extract_psg_data(
            participant_raw_psg_cardiac_data=participant_raw_psg_cardiac_data,
            participant_raw_psg_respiratory_data=participant_raw_psg_respiratory_data,
            participant_processed_data=participant_processed_data,
        )

    # Write processed file
    os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)
    with h5py.File(processed_file_path, "w") as f:
        _dict_to_h5py_group(participant_processed_data, f)


def preprocess_db2() -> None:
    """
    Preprocess all DB2 participant raw folders.

    This function retrieves all raw DB2 participant folders and processes
    each one sequentially with a progress bar.
    """
    participant_raw_folder_paths = get_db2_participant_raw_folder_paths()
    print(f"ℹ️  Found {len(participant_raw_folder_paths)} participant raw folders.")
    for folder_path in tqdm(
        participant_raw_folder_paths, desc="⌛ Processing DB2 data..."
    ):
        _process_participant_raw_folder(folder_path)
