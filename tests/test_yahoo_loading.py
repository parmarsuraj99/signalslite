import tempfile
import unittest
import numpy as np
import pandas as pd
import requests
import os
import multiprocessing

from signalslite.constants import Directories
from signalslite.data_utils import save_in_folders
from signalslite.data_downloader import update_daily_data, StockDataDownloader


class TestSaving(unittest.TestCase):
    """
    Tests if yahoo data is downloadable, just for few tickers
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_name = self.temp_dir.name
        self.dir_config = Directories()
        self.dir_config.set_data_dir(self.temp_dir_name)

    def download_ticker_map(self):
        data_dir = self.dir_config.DATA_DIR
        ticker_map_fname = f"{data_dir}/eodhd-map.csv"
        if not os.path.exists(ticker_map_fname):
            print(f"Missing ticker map file: {ticker_map_fname}")
            url = "https://raw.githubusercontent.com/parmarsuraj99/dsignals/main/db/eodhd-map.csv"
            r = requests.get(url)
            if r.status_code == requests.codes.ok:
                with open(ticker_map_fname, "wb") as f:
                    f.write(r.content)
        # filter ticker map for yahoo tickers
        ticker_map = pd.read_csv(ticker_map_fname).set_index("bloomberg_ticker")
        yahoo_tickers = (
            ticker_map[ticker_map["data_provider"] == "yahoo"].dropna().head(20)
        )
        print(yahoo_tickers)

        # overwrite ticker map with yahoo tickers
        yahoo_tickers.to_csv(f"{data_dir}/eodhd-map.csv")

    def test_yahoo_loading(self):

        # test on a smaller universe; 20 tickers
        self.download_ticker_map()

        downloader = StockDataDownloader(
            max_workers=multiprocessing.cpu_count() - 1,
        )

        ticker_map = pd.read_csv(f"{self.dir_config.DATA_DIR}/eodhd-map.csv").set_index(
            "bloomberg_ticker"
        )
        latest_date = "2022-01-01"

        all_quotes = downloader.download_all(ticker_map, date_from=latest_date)
        all_quotes = [quotes for quotes in all_quotes if len(quotes) > 0]
        all_quotes = pd.concat(all_quotes, join="outer", axis=0)
        print(all_quotes["bloomberg_ticker"].unique(), all_quotes.shape)

        print(all_quotes.head())

        save_in_folders(all_quotes, self.dir_config.DAILY_DATA_DIR)

        # assert
        self.assertTrue(self.dir_config.DAILY_DATA_DIR.is_dir())

        # length of dates in the directory

    def tearDown(self):
        self.temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
