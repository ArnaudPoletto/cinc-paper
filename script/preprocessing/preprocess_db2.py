"""
Script to preprocess DB2 dataset participant data.

This script serves as an entry point to trigger the DB2 preprocessing pipeline,
which converts raw TDMS files to HDF5 format with processed signals.

Authors: Arnaud Poletto
"""

from cinc.preprocessing.preprocess_db2 import preprocess_db2


def main() -> None:
    """
    Execute the DB2 preprocessing pipeline.

    This function calls the main preprocessing function that processes all
    DB2 participant raw folders and converts them to HDF5 format.
    """
    print("▶️  Starting DB2 data processing.")
    preprocess_db2()
    print("✅ DB2 data processing completed!")


if __name__ == "__main__":
    main()
