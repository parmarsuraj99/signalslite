import gc
import logging
import multiprocessing
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from .base import StorageReader, StorageWriter


class LocalStorageWriter(StorageWriter):
    def __init__(self, root_dir: str):
        super().__init__(root_dir)

    @staticmethod
    def save_daily_data(
        date: str, df_slice: pd.DataFrame, level_path: str, overwrite: bool = False
    ):
        """
        Save the daily data to the given directory.
        params:
            date: str
                The date in the format YYYY-MM-DD.
            df_slice: pd.DataFrame
                The DataFrame containing the daily data.
            level_path: str
                The path to the directory to save the data to.
        """
        year, month, day = date.split("-")
        level_path = Path(level_path, year, month)
        level_path.mkdir(parents=True, exist_ok=True)
        fname = os.path.join(level_path, f"{date}.parquet")
        if not os.path.exists(fname):
            df_slice.to_parquet(fname)

    def write_ohlcv(
        self, df: pd.DataFrame, subdir: str, force: bool = False, reset: bool = False
    ):
        """
        Root dir: `data`
        subdir: `ohlcv`
        Saves under: `data/ohlcv/<year>/<month>/<date>.parquet`

        parms:
            df: pd.DataFrame
                The DataFrame to write. Must contain the following columns:
                - ticker: str
                    The ticker of the stock.
                - date_str: str
                    The date in the format YYYY-MM-DD.
                - open, high, low, close: float
                    The OHLC values.
            subdir: str
                The subdirectory to write the data to.
            force: bool
                Whether to overwrite the existing data. placeholder for now.
            reset: bool
                Whether to reset the directory before writing.
        """

        assert (
            "date_str" in df.columns
        ), "The DataFrame must contain the column date_str."

        if reset:
            # logging.info(f"Resetting {subdir} directory.")
            # # this is os agnostic
            # os.remove(self.root_dir / subdir)
            # os.mkdir(self.root_dir / subdir)
            pass

        logging.info(f"cpu_count: {multiprocessing.cpu_count()}")
        Parallel(n_jobs=multiprocessing.cpu_count() - 1)(
            delayed(self.save_daily_data)(date, df_slice, self.root_dir / subdir)
            for date, df_slice in tqdm(df.groupby("date_str"))
        )
        gc.collect()

    def write_fundamentals(self, ticker_to_fundamentals: Dict[str, Any], subdir: str):
        """
        Save the fundamentals data to the given directory.
        params:
            ticker_to_fundamentals: Dict[str, Any]
                A dictionary mapping the ticker symbol to the fundamentals data.
            subdir: str
                The subdirectory to write the data to.
        """
        today_date_utc = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        # create the directory if it doesn't exist
        (self.root_dir / subdir).mkdir(parents=True, exist_ok=True)
        for ticker, fundamentals in ticker_to_fundamentals.items():
            ticker = ticker.replace("/", "__")
            pd.to_pickle(
                fundamentals, self.root_dir / subdir / f"{ticker}_{today_date_utc}.pkl"
            )


class LocalStorageReader(StorageReader):
    def __init__(self, root_dir: str):
        super().__init__(root_dir)

    def read_available_dates_ohlcv(self, subdir: Path, extension: str = ".parquet"):
        """
        Read all available dates from the given directory.
        params:
            dir: Path
                The directory to read the dates from.
        returns:
            List[str]
                A list of dates in the format YYYY-MM-DD.
        """
        ohlcv_dir = self.root_dir / subdir

        if not os.path.exists(ohlcv_dir):
            return []

        return sorted(
            [
                file[: -len(extension)]
                for root, dirs, files in os.walk(ohlcv_dir)
                for file in files
                if file.endswith(".parquet")
            ]
        )

    def load_last_n_days_ohlcv(
        self, subdir: str, last_n_days: int = -1
    ) -> Optional[pd.DataFrame]:
        """
        Read the last n days of OHLCV data.
        params:
            n_days: int
                The number of days to read.
        returns:
            Optional[pd.DataFrame]
                A DataFrame containing the OHLCV data.
        """
        dates = self.read_available_dates_ohlcv(subdir=subdir)
        dates = dates[-last_n_days:]

        df = pd.concat(
            [pd.read_parquet(self.root_dir / f"{date}.parquet") for date in dates]
        )
        return df


class LocalStorage(LocalStorageReader, LocalStorageWriter):
    def __init__(self, root_dir: str):
        super().__init__(root_dir)
