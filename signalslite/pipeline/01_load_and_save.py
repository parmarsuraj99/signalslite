import logging
import os

import pandas as pd

from signalslite.downloaders import stock_downloader
from signalslite.storages import local_storage


class RawDataPipeline:
    OHLCV_DIR = "ohlcv"
    FUNDAMENTALS_DIR = "fundamentals"

    def __init__(
        self,
        root_dir: str,
        storage_interface,
        mappig: pd.DataFrame,
        EODHD_API_KEY: str,
    ):
        self.EODHD_API_KEY = EODHD_API_KEY
        self.mappig = mappig

        self.root_dir = root_dir
        self.storage_interface = storage_interface
        self.ohlcv_downloader = stock_downloader.OHLCVDownloader(EODHD_API_KEY, mappig)
        self.fundamentals_downloader = stock_downloader.StockFundamentalDownloader(
            EODHD_API_KEY, mappig
        )

    def download_ohlcv(self, from_date: str = None, date_until: str = None):
        """
        Download the OHLCV data for all tickers in the mapping file.
        """
        logging.info("Downloading OHLCV data.")
        all_quotes = self.ohlcv_downloader.download_all(
            date_from=from_date, date_until=date_until
        )
        all_quotes = pd.concat(all_quotes)
        all_quotes.index = pd.to_datetime(all_quotes.index, format="%Y-%m-%d")
        all_quotes["date_str"] = all_quotes.index.strftime("%Y-%m-%d")
        # log column names
        logging.info(f"columns: {all_quotes.columns}")
        return all_quotes

    def update_ohlcv(self):
        """
        Update the OHLCV data for all tickers in the mapping file.
        """
        available_dates = self.storage_interface.read_available_dates_ohlcv(
            self.OHLCV_DIR
        )
        date_today_utc = pd.Timestamp.utcnow().strftime("%Y-%m-%d")

        if len(available_dates) == 0:
            logging.info("No data available yet.")
            all_quotes = self.download_ohlcv()
            self.storage_interface.write_ohlcv(all_quotes, self.OHLCV_DIR)

        else:
            last_date = available_dates[
                -1
            ]  # should be -1 but just for robustness loading another day since it'll be overwritten anyway
            logging.info(f"last_date: {last_date}")
            if last_date == date_today_utc:
                logging.info("Already up to date.")
            else:
                logging.info(f"last_date: {last_date}")
                all_quotes = self.download_ohlcv(from_date=last_date)
                self.storage_interface.write_ohlcv(
                    all_quotes, self.OHLCV_DIR, reset=True
                )

    def download_fundamentals(self):
        """
        Download the fundamentals data for all tickers in the mapping file.
        """
        logging.info("Downloading fundamentals data.")
        all_fundamentals = self.fundamentals_downloader.download_all(self.mappig)
        return all_fundamentals

    def update_fundamentals(self):
        """
        Update the fundamentals data for all tickers in the mapping file.
        """
        logging.info("Updating fundamentals data.")
        # fetach latest fundamental data
        all_fundamentals = self.download_fundamentals()

        # write to storage
        self.storage_interface.write_fundamentals(
            all_fundamentals, self.FUNDAMENTALS_DIR
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    mappig = pd.read_csv("data/eodhd-map.csv").set_index("bloomberg_ticker")
    EODHD_API_KEY = os.environ.get("EODHD_API_KEY")

    assert (
        EODHD_API_KEY is not None and len(EODHD_API_KEY) > 10
    ), "Please set the EODHD_API_KEY environment variable."

    storage_interface = local_storage.LocalStorage("data")
    pipeline = RawDataPipeline("data", storage_interface, mappig, EODHD_API_KEY)
    pipeline.update_ohlcv()

    # update fundamentals
    pipeline.update_fundamentals()
