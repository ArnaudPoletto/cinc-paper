"""
This utility file provides functions to work with HDF5 files using h5py.

Authors: Arnaud Poletto
"""

import h5py
import numpy as np
from typing import Dict


def _dict_to_h5py_group(data: Dict, group: h5py.Group) -> None:
    """
    Recursively convert a dictionary to an h5py group.

    Args:
        data (Dict): The dictionary to convert.
        group (h5py.Group): The h5py group to populate with the data.
    """
    for key, value in data.items():
        if value is None:
            continue

        if isinstance(value, dict):
            sub_group = group.create_group(key)
            _dict_to_h5py_group(value, sub_group)

        elif isinstance(value, str):
            dt = h5py.string_dtype(encoding="utf-8")
            group.create_dataset(key, data=value, dtype=dt)

        elif isinstance(value, list):
            if len(value) == 0:
                group.create_dataset(key, data=np.empty((0,)))
            elif isinstance(value[0], np.ndarray):
                list_group = group.create_group(key)
                for i, array in enumerate(value):
                    list_group.create_dataset(str(i), data=array, compression="gzip")
            else:
                arr = np.asarray(value)
                group.create_dataset(key, data=arr, compression="gzip")

        elif np.isscalar(value) or (
            isinstance(value, np.ndarray) and value.shape == ()
        ):
            group.create_dataset(key, data=value)

        else:
            group.create_dataset(key, data=value, compression="gzip")
