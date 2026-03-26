"""
This utility file provides functions to retrieve file paths for participant data.

It supports accessing raw and processed data files across multiple datasets with
validation and error handling.

Authors: Arnaud Poletto
"""

from typing import List

from cinc.utils.filesystem import get_paths_recursive
from cinc.config import (
    RAW_DB2_PATH,
    PROCESSED_DB2_PATH,
    PROCESSED_DB2_PARTS_PATH,
)


# ============================================================================
# DB2 Dataset Functions
# ============================================================================


def get_db2_participant_raw_folder_paths() -> List[str]:
    """
    Get the paths of all participant raw folders in the DB2 dataset.

    Returns:
        folder_paths (List[str]): The list of paths to the participant raw folders.
    """
    return get_paths_recursive(
        folder_path=RAW_DB2_PATH,
        match_pattern="*",
        path_type="d",
        recursive=False,
    )


def get_db2_participant_processed_file_paths() -> List[str]:
    """
    Get the paths of all processed participant files in the DB2 dataset.

    Returns:
        file_paths (List[str]): The list of paths to the processed participant files.
    """
    return get_paths_recursive(
        folder_path=PROCESSED_DB2_PATH,
        match_pattern="*.h5",
        path_type="f",
        recursive=True,
    )


# ============================================================================
# DB2 Parts Dataset Functions
# ============================================================================


def get_db2_parts_participant_processed_file_paths() -> List[str]:
    """
    Get the paths of all processed participant files in the DB2 Parts dataset.

    Returns:
        file_paths (List[str]): The list of paths to the processed participant files.
    """
    return get_paths_recursive(
        folder_path=PROCESSED_DB2_PARTS_PATH,
        match_pattern="*.h5",
        path_type="f",
        recursive=True,
    )


def get_db2_parts_participant_file_path_from_file_name(file_name: str) -> str:
    """
    Get the full path of a participant file in the DB2 Parts dataset based on the file name.

    Args:
        file_name (str): The name of the participant file.

    Returns:
        file_path (str): The full path to the participant file.

    Raises:
        FileNotFoundError: If the participant file is not found.
    """
    participant_file_paths = get_paths_recursive(
        folder_path=PROCESSED_DB2_PARTS_PATH,
        match_pattern=f"{file_name}.h5",
        path_type="f",
        recursive=True,
    )

    if not participant_file_paths:
        raise FileNotFoundError(
            f"❌ Participant file '{file_name}' not found in DB2 Parts dataset."
        )

    return participant_file_paths[0]


def check_db2_parts_participant_objects_exist() -> None:
    """
    Check if the DB2 Parts participant raw folders and processed files exist.

    Raises:
        FileNotFoundError: If no processed files are found.
    """
    processed_files = get_db2_parts_participant_processed_file_paths()

    # Check if processed files exist
    if not processed_files:
        raise FileNotFoundError(
            "❌ No processed participant files found in DB2 Parts dataset. Please run the processing step first."
        )
