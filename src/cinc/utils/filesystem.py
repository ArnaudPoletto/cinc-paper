"""
This utility file provides functions to interact with the filesystem, such as retrieving file paths based on specific patterns and criteria.

Authors: Arnaud Poletto
"""

from typing import List, Optional
from pathlib import Path


def get_paths_recursive(
    folder_path: str,
    match_pattern: str,
    path_type: Optional[str] = None,
    recursive: bool = True,
) -> List[str]:
    """
    Get all file paths in the given folder path that match the given pattern. Paths can be filtered by type (file or directory) and can be searched recursively.

    Args:
        folder_path (str): The path to the folder.
        match_pattern (str): The pattern to match the file names.
        path_type (Optional[str], optional): The type of path to return. Must be None, 'f', or 'd'. Defaults to None.
        recursive (bool, optional): Whether to search recursively. Defaults to True.

    Raises:
        ValueError: If an invalid path type is provided.

    Returns:
        paths (List[str]): The list of file paths that match the given pattern.
    """
    if path_type not in [None, "f", "d"]:
        raise ValueError(
            f"❌ Invalid file type {path_type}. Must be None, 'f', or 'd'."
        )

    # Define search method and get paths
    search_method = Path(folder_path).rglob if recursive else Path(folder_path).glob
    paths = list(search_method(match_pattern))

    # Filter and resolve paths
    paths = [
        path.resolve().as_posix()
        for path in paths
        if (
            path_type is None
            or (path_type == "f" and path.is_file())
            or (path_type == "d" and path.is_dir())
        )
    ]

    return paths
