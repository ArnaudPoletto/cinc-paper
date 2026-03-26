"""
This utility file provides functions to load participant data from HDF5 files.

It supports loading preprocessed data along with optional detection results
(cardiac and respiratory) from both polysomnography and piezoelectric sensors.

Authors: Arnaud Poletto
"""

import os
import h5py
from typing import Dict


def _h5py_to_dict(item) -> Dict:
    """
    Recursively convert an HDF5 group or dataset to a Python dictionary.

    Args:
        item: An h5py.Group or h5py.Dataset object.

    Returns:
        result (Dict): A dictionary representation of the HDF5 structure.
    """
    if isinstance(item, h5py.Group):
        return {key: _h5py_to_dict(item[key]) for key in item.keys()}
    else:
        return item[()]


def _get_signal_shape(participant_file_path: str, *keys: str):
    """
    Read just the shape of a signal dataset from the processed HDF5 file
    without loading the data into memory.

    Args:
        participant_file_path (str): The path to the participant's processed data file.
        *keys (str): The nested keys to traverse to reach the signal dataset
            (e.g., "pel", "cardiac", "processed", "signal").

    Returns:
        shape (tuple): The shape of the signal dataset, or None if the path does not exist.
    """
    with h5py.File(participant_file_path, "r") as file:
        item = file
        for key in keys:
            if key not in item:
                return None
            item = item[key]
        return item.shape


def _load_psg_cardiac_detection(
    participant_file_path: str,
    participant_data: Dict,
) -> None:
    """
    Load polysomnography cardiac detection results and add them to participant data.

    Args:
        participant_file_path (str): The path to the participant's processed data file.
        participant_data (Dict): The participant data dictionary to update in-place.
    """
    psg_cardiac_detection_file_path = participant_file_path.replace(
        "processed", "cardiac_detection"
    ).replace(".h5", "_psg_cardiac.h5")

    if os.path.exists(psg_cardiac_detection_file_path):
        with h5py.File(psg_cardiac_detection_file_path, "r") as file:
            psg_cardiac_detection_data = _h5py_to_dict(file)
            psg_dict = participant_data.setdefault("psg", {})
            cardiac_dict = psg_dict.setdefault("cardiac", {})
            cardiac_dict["detection"] = {"channel_0": psg_cardiac_detection_data}


def _load_pel_cardiac_detection(
    participant_file_path: str,
    participant_data: Dict,
) -> None:
    """
    Load piezoelectric cardiac detection results and add them to participant data.

    Args:
        participant_file_path (str): The path to the participant's processed data file.
        participant_data (Dict): The participant data dictionary to update in-place.
    """
    # Check if piezoelectric cardiac data exists
    signal_shape = _get_signal_shape(
        participant_file_path, "pel", "cardiac", "processed", "signal"
    )
    if signal_shape is None:
        return

    # Initialize detection dictionary
    cardiac_dict = participant_data.setdefault("pel", {}).setdefault("cardiac", {})
    detection_dict = cardiac_dict.setdefault("detection", {})

    # Load individual sensor detection
    n_sensors = signal_shape[0]
    for i in range(n_sensors):
        pel_cardiac_detection_file_path = participant_file_path.replace(
            "processed", "cardiac_detection"
        ).replace(".h5", f"_pel_cardiac_{i}.h5")

        if os.path.exists(pel_cardiac_detection_file_path):
            with h5py.File(pel_cardiac_detection_file_path, "r") as file:
                pel_cardiac_detection_data = _h5py_to_dict(file)
                detection_dict[f"sensor_{i}"] = pel_cardiac_detection_data

    # Load ensemble detection
    pel_cardiac_detection_file_path = participant_file_path.replace(
        "processed", "cardiac_detection"
    ).replace(".h5", "_pel_cardiac_ensemble.h5")

    if os.path.exists(pel_cardiac_detection_file_path):
        with h5py.File(pel_cardiac_detection_file_path, "r") as file:
            pel_cardiac_detection_data = _h5py_to_dict(file)
            detection_dict["ensemble"] = pel_cardiac_detection_data


def _load_psg_respiratory_detection(
    participant_file_path: str,
    participant_data: Dict,
) -> None:
    """
    Load polysomnography respiratory detection results and add them to participant data.

    Args:
        participant_file_path (str): The path to the participant's processed data file.
        participant_data (Dict): The participant data dictionary to update in-place.
    """
    # Check if polysomnography respiratory data exists
    signal_shape = _get_signal_shape(
        participant_file_path, "psg", "respiratory", "processed", "signal"
    )
    if signal_shape is None:
        return

    # Initialize detection dictionary
    respiratory_dict = participant_data.setdefault("psg", {}).setdefault("respiratory", {})
    detection_dict = respiratory_dict.setdefault("detection", {})

    # Load detection based on number of channels
    if len(signal_shape) == 1:
        # Handle single-channel case
        psg_respiratory_detection_file_path = participant_file_path.replace(
            "processed", "respiratory_detection"
        ).replace(".h5", "_psg_respiratory.h5")

        if os.path.exists(psg_respiratory_detection_file_path):
            with h5py.File(psg_respiratory_detection_file_path, "r") as file:
                psg_respiratory_detection_data = _h5py_to_dict(file)
                detection_dict["channel_0"] = psg_respiratory_detection_data
    else:
        # Handle multi-channel case
        n_channels = signal_shape[0]
        for i in range(n_channels):
            psg_respiratory_detection_file_path = participant_file_path.replace(
                "processed", "respiratory_detection"
            ).replace(".h5", f"_psg_respiratory_{i}.h5")

            if os.path.exists(psg_respiratory_detection_file_path):
                with h5py.File(psg_respiratory_detection_file_path, "r") as file:
                    psg_respiratory_detection_data = _h5py_to_dict(file)
                    detection_dict[f"channel_{i}"] = psg_respiratory_detection_data


def _load_pel_respiratory_detection(
    participant_file_path: str,
    participant_data: Dict,
) -> None:
    """
    Load piezoelectric respiratory detection results and add them to participant data.

    Args:
        participant_file_path (str): The path to the participant's processed data file.
        participant_data (Dict): The participant data dictionary to update in-place.
    """
    # Check if piezoelectric respiratory data exists
    signal_shape = _get_signal_shape(
        participant_file_path, "pel", "respiratory", "processed", "signal"
    )
    if signal_shape is None:
        return

    # Initialize detection dictionary
    respiratory_dict = participant_data.setdefault("pel", {}).setdefault("respiratory", {})
    detection_dict = respiratory_dict.setdefault("detection", {})

    # Load individual sensor detections
    n_sensors = signal_shape[0]
    for i in range(n_sensors):
        pel_respiratory_detection_file_path = participant_file_path.replace(
            "processed", "respiratory_detection"
        ).replace(".h5", f"_pel_respiratory_{i}.h5")

        if os.path.exists(pel_respiratory_detection_file_path):
            with h5py.File(pel_respiratory_detection_file_path, "r") as file:
                pel_respiratory_detection_data = _h5py_to_dict(file)
                detection_dict[f"sensor_{i}"] = pel_respiratory_detection_data

    # Load ensemble detection
    pel_respiratory_detection_file_path = participant_file_path.replace(
        "processed", "respiratory_detection"
    ).replace(".h5", "_pel_respiratory_ensemble.h5")

    if os.path.exists(pel_respiratory_detection_file_path):
        with h5py.File(pel_respiratory_detection_file_path, "r") as file:
            pel_respiratory_detection_data = _h5py_to_dict(file)
            detection_dict["ensemble"] = pel_respiratory_detection_data


def _load_feature(
    participant_file_path: str,
    participant_data: Dict,
    sensor_type: str,
    category: str,
    feature: str,
) -> None:
    """
    Load a specific feature result and add it to participant data.

    Args:
        participant_file_path (str): The path to the participant's processed data file.
        participant_data (Dict): The participant data dictionary to update in-place.
        sensor_type (str): The type of sensor ("psg" or "pel").
        category (str): The category to store the feature in (e.g., "cardiac", "respiratory", "raw"). Defaults to "cardiac".
        feature (str): The name of the feature to load.
    """
    parts = participant_file_path.rsplit("/processed/", 1)
    feature_file_path = (f"{parts[0]}/features/{feature}/{parts[1]}").replace(
        ".h5", f"_{sensor_type}.h5"
    )

    if not os.path.exists(feature_file_path):
        return

    with h5py.File(feature_file_path, "r") as file:
        feature_data = _h5py_to_dict(file)
        sensor_dict = participant_data.setdefault(sensor_type, {})
        category_dict = sensor_dict.setdefault(category, {})
        features_dict = category_dict.setdefault("features", {})
        features_dict[feature] = feature_data


def _load_psg_features(
    participant_file_path: str,
    participant_data: Dict,
) -> None:
    """
    Load PSG feature results and add them to participant data.

    Args:
        participant_file_path (str): The path to the participant's processed data file.
        participant_data (Dict): The participant data dictionary to update in-place.
    """
    # Load cardiac features
    cardiac_features = [
        "cardiac_dmhr",
        "cardiac_mhr",
        "cardiac_misint",
        "cardiac_rmssd",
        "cardiac_sd1",
        "cardiac_sd12",
        "cardiac_sdnn",
        "movement_energy", # From cardiac signal for PSG
    ]
    for feature in cardiac_features:
        _load_feature(participant_file_path, participant_data, "psg", "cardiac", feature)

def _load_pel_features(
    participant_file_path: str,
    participant_data: Dict,
) -> None:
    """
    Load piezoelectric feature results and add them to participant data.

    Args:
        participant_file_path (str): The path to the participant's processed data file.
        participant_data (Dict): The participant data dictionary to update in-place.
    """
    # Load cardiac features
    cardiac_features = [
        "cardiac_dmhr",
        "cardiac_mhr",
        "cardiac_misint",
        "cardiac_rmssd",
        "cardiac_sd1",
        "cardiac_sd12",
        "cardiac_sdnn",
    ]
    for feature in cardiac_features:
        _load_feature(participant_file_path, participant_data, "pel", "cardiac", feature)

    raw_features = [
        "movement_energy",
    ]
    for feature in raw_features:
        _load_feature(participant_file_path, participant_data, "pel", "raw", feature)


def _load_pre_respiratory_detection(
    participant_file_path: str,
    participant_data: Dict,
) -> None:
    """
    Load piezoresistive respiratory detection results and add them to participant data.

    Args:
        participant_file_path (str): The path to the participant's processed data file.
        participant_data (Dict): The participant data dictionary to update in-place.
    """
    # Check if piezoresistive respiratory data exists
    signal_shape = _get_signal_shape(
        participant_file_path, "pre", "respiratory", "processed", "signal"
    )
    if signal_shape is None:
        return

    # Initialize detection dictionary
    respiratory_dict = participant_data.setdefault("pre", {}).setdefault("respiratory", {})
    detection_dict = respiratory_dict.setdefault("detection", {})

    # Load individual sensor detections
    n_sensors = signal_shape[0]
    for i in range(n_sensors):
        pre_respiratory_detection_file_path = participant_file_path.replace(
            "processed", "respiratory_detection"
        ).replace(".h5", f"_pre_respiratory_{i}.h5")

        if os.path.exists(pre_respiratory_detection_file_path):
            with h5py.File(pre_respiratory_detection_file_path, "r") as file:
                pre_respiratory_detection_data = _h5py_to_dict(file)
                detection_dict[f"sensor_{i}"] = pre_respiratory_detection_data

    # Load ensemble detection
    pre_respiratory_detection_file_path = participant_file_path.replace(
        "processed", "respiratory_detection"
    ).replace(".h5", "_pre_respiratory_ensemble.h5")

    if os.path.exists(pre_respiratory_detection_file_path):
        with h5py.File(pre_respiratory_detection_file_path, "r") as file:
            pre_respiratory_detection_data = _h5py_to_dict(file)
            detection_dict["ensemble"] = pre_respiratory_detection_data


def get_participant_data(
    participant_file_path: str,
    with_psg_cardiac_detection: bool = False,
    with_psg_respiratory_detection: bool = False,
    with_psg_features: bool = False,
    with_pel_cardiac_detection: bool = False,
    with_pel_respiratory_detection: bool = False,
    with_pel_features: bool = False,
    with_pre_respiratory_detection: bool = False,
    with_base_data: bool = True,
) -> Dict:
    """
    Load participant data from an HDF5 file with optional detection results.

    This function loads the base participant data from a processed HDF5 file
    and optionally merges cardiac and respiratory detection results from
    separate detection files.

    Args:
        participant_file_path (str): The path to the participant's processed data file.
        with_psg_cardiac_detection (bool, optional): Whether to load PSG cardiac detection results. Defaults to False.
        with_psg_respiratory_detection (bool, optional): Whether to load PSG respiratory detection results. Defaults to False.
        with_psg_features (bool, optional): Whether to load PSG feature results. Defaults to False.
        with_pel_cardiac_detection (bool, optional): Whether to load piezoelectric cardiac detection results. Defaults to False.
        with_pel_respiratory_detection (bool, optional): Whether to load piezoelectric respiratory detection results. Defaults to False.
        with_pel_features (bool, optional): Whether to load piezoelectric feature results. Defaults to False.
        with_pre_respiratory_detection (bool, optional): Whether to load piezoresistive respiratory detection results. Defaults to False.
        with_base_data (bool, optional): Whether to load the base participant data from the processed file. Defaults to True.
            Note: some detection loaders (pel_cardiac, psg/pel/pre_respiratory) depend on base data
            to determine sensor/channel counts. They will be skipped if base data is not loaded.

    Returns:
        participant_data (Dict): The nested dictionary containing the participant's data.
    """
    # Load base participant data
    if with_base_data:
        with h5py.File(participant_file_path, "r") as file:
            participant_data = _h5py_to_dict(file)
    else:
        participant_data = {}

    # Load optional detection results
    if with_psg_cardiac_detection:
        _load_psg_cardiac_detection(participant_file_path, participant_data)

    if with_psg_respiratory_detection:
        _load_psg_respiratory_detection(participant_file_path, participant_data)

    if with_psg_features:
        _load_psg_features(participant_file_path, participant_data)

    if with_pel_cardiac_detection:
        _load_pel_cardiac_detection(participant_file_path, participant_data)

    if with_pel_respiratory_detection:
        _load_pel_respiratory_detection(participant_file_path, participant_data)

    if with_pel_features:
        _load_pel_features(participant_file_path, participant_data)

    if with_pre_respiratory_detection:
        _load_pre_respiratory_detection(participant_file_path, participant_data)

    return participant_data
