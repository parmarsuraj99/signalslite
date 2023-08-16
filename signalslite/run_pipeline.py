from signalslite.data_downloader import update_daily_data
from signalslite.primary_features import generate_primary_features
from signalslite.secondary_features import update_secondary_features
from signalslite.scale_data import scale_data
from signalslite.merge_targets import prepare_merged_data
from signalslite.constants import Directories
from credentials import EODHD_API_KEY

if __name__ == "__main__":
    dir_config = Directories()
    dir_config.set_data_dir("data")

    EODHD_API_KEY = None

    print(dir_config)
    update_daily_data(dir_config.DATA_DIR, dir_config.DAILY_DATA_DIR, EODHD_API_KEY)
    generate_primary_features(dir_config)
    update_secondary_features(dir_config)
    scale_data(dir_config, FROM_SCRATCH=False)
    prepare_merged_data(dir_config)
