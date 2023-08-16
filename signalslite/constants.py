import os
import dataclasses
from pathlib import Path

# TODO: Load this separately in all files and allow user to change it; potentially for cloud storage facility
# TODO: How about allowing users to define their own class for data_utils.py? (e.g. for cloud storage)
# But first, get this working with the current setup; start on cloud \
# after V0.0.1 in cloud_branch; constraints stays same for efficiency

# use dataclass for constants
class Directories:
    DATA_DIR = Path("data")
    DAILY_DATA_DIR = DATA_DIR / "01_daily_adjusted"
    DAILY_PRIMARY_FEATURES_DIR = DATA_DIR / "02_primary_features"
    DAILY_SECONDARY_FEATURES_DIR = DATA_DIR / "03_secondary_features"
    DAILY_SCALED_FEATURES_DIR = DATA_DIR / "04_scaled_features"

    @classmethod
    def set_data_dir(cls, new_data_dir):
        cls.DATA_DIR = Path(new_data_dir)
        cls.DAILY_DATA_DIR = cls.DATA_DIR / "01_daily_adjusted"
        cls.DAILY_PRIMARY_FEATURES_DIR = cls.DATA_DIR / "02_primary_features"
        cls.DAILY_SECONDARY_FEATURES_DIR = cls.DATA_DIR / "03_secondary_features"
        cls.DAILY_SCALED_FEATURES_DIR = cls.DATA_DIR / "04_scaled_features"
