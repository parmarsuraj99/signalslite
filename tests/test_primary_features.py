import tempfile
import unittest
import numpy as np
import pandas as pd

from signalslite.data_utils import load_recent_data_from_file, read_available_dates, save_in_folders
from signalslite.constants import Directories
from signalslite.primary_features import generate_primary_features, 

class TestSaving(unittest.TestCase):
    """
    create a temporary directory and save some data there, tests primary feature_generation
    """
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_name = self.temp_dir.name
        self.dir_config = Directories()
        self.dir_config.set_data_dir(self.temp_dir_name)
        self.fake_data = self.generate_fake_data()
        print(self.fake_data.columns)

    def generate_fake_data(self):
        # generate 1000 days of stock market ohlcv data for 10 stocks
        n_days = 1000
        n_stocks = 1000
        dates = pd.date_range('2010-01-01', periods=n_days, freq='D')

        tickers = [f'STOCK_{i}' for i in range(n_stocks)]

        full_df = []
        for i in range(n_stocks):
            data = np.random.randint(1, 100, (n_days, 6))
            data = pd.DataFrame(data, index=dates, columns=["open", "high", "low", "close", "adjusted_close", "volume"])
            data["bloomberg_ticker"] = tickers[i]

            full_df.append(data)
        
        full_df = pd.concat(full_df)

        return full_df
    
    def test_primary_feature_generation(self):
        save_in_folders(self.fake_data, self.dir_config.DAILY_DATA_DIR)
        available_dates = read_available_dates(self.dir_config.DAILY_DATA_DIR)
        recent_data = load_recent_data_from_file(self.dir_config.DAILY_DATA_DIR, n_days=-1)
        unique_dates = recent_data.index.unique()
        self.assertEqual(len(unique_dates), 1000)
        generate_primary_features(self.dir_config)
        self.assertTrue(self.dir_config.DAILY_PRIMARY_FEATURES_DIR.exists())
        self.assertTrue(self.dir_config.DAILY_PRIMARY_FEATURES_DIR.is_dir())

        # assert length of primary features is 1000
        primary_features_dates = read_available_dates(self.dir_config.DAILY_PRIMARY_FEATURES_DIR)

        # max window is 200; so initial values will be nan
        self.assertEqual(len(primary_features_dates), 1000 - 200)


    def tearDown(self):
        self.temp_dir.cleanup()

if __name__ == '__main__':
    unittest.main()