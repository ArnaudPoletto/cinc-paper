import numpy as np
from pathlib import Path

# Paths
try:
    from config_local import DATA_PATH
except ImportError:
    CONFIG_DIR = Path(__file__).parent
    DATA_PATH = CONFIG_DIR / "../../data"

RAW_DATA_PATH = f"{DATA_PATH}/raw"
PROCESSED_DATA_PATH = f"{DATA_PATH}/processed"

RAW_DB2_PATH = f"{RAW_DATA_PATH}/db2"
PROCESSED_DB2_PATH = f"{PROCESSED_DATA_PATH}/db2"
PROCESSED_DB2_PARTS_PATH = f"{PROCESSED_DATA_PATH}/db2_parts"

FEATURES_PATH = f"{DATA_PATH}/features"

# Processsing
PREPROCESSING_CLEAN_THRESHOLD_STD = 1.0
PROCESSING_VARIABLES = {
    "psg": {
        "cardiac": {
            "lowcut": 0.5,
            "highcut": 40.0,
            "processed_fs": 100.0,
            "upsampled_fs": 1000.0,
        },
        "respiratory": {
            "processed_fs": 100.0,
        },

    },
    "pel": {
        "raw_fs": 100.0,
        "cardiac": {
            "lowcut": 1,
            "highcut": 10,
            "processed_fs": 100.0,
            "upsampled_fs": 1000.0,
        },
        "respiratory": {
            "processed_fs": 100.0,
        },
    },
    "pre": {
        "respiratory": {
            "processed_fs": 100.0,
        }
    }
}

DB2_N_PEL_SENSORS = 16 
DB2_N_PRE_SENSORS = 16
DB2_PRE_SENSOR_NAMES = [
    "Res_1LC", "Res_1RC", "Res_1LL", "Res_1RL",
    "Res_2LC", "Res_2RC", "Res_2LL", "Res_2RL",
    "Res_3LC", "Res_3RC", "Res_3LL", "Res_3RL",
    "Res_4LC", "Res_4RC", "Res_4LL", "Res_4RL",
]

DB2_PART_MARGIN = 30
DB2_PART_DURATION = 180
DB2_PARTS_S = [
    [DB2_PART_MARGIN, DB2_PART_MARGIN + DB2_PART_DURATION],
    [3 * DB2_PART_MARGIN + DB2_PART_DURATION, 3 * DB2_PART_MARGIN + 2 * DB2_PART_DURATION],
    [5 * DB2_PART_MARGIN + 2 * DB2_PART_DURATION, 5 * DB2_PART_MARGIN + 3 * DB2_PART_DURATION],
    [7 * DB2_PART_MARGIN + 3 * DB2_PART_DURATION, 7 * DB2_PART_MARGIN + 4 * DB2_PART_DURATION],
]
DB2_PART_NAMES = ["Prone", "Right", "Left", "Supine"]