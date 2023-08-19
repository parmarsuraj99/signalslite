import os
import gc
import time
from pathlib import Path
import requests
from datetime import datetime
import pandas as pd
from concurrent import futures
from tqdm import tqdm

from joblib import Parallel, delayed
import multiprocessing

from signalslite.constants import Directories

from signalslite.data_utils import (
    load_recent_data_from_file,
    save_daily_data,
    get_latest_date,
    save_in_folders,
)


class YahooDownloaderOHLCV:
    def __init__(self):
        pass

    @staticmethod
    def _download_dividends_yahoo(ticker: str, start_date: str, end_date: str):
        start_epoch = int(start_date.timestamp())
        end_epoch = int(end_date.timestamp())

        dividends = (
            pd.read_csv(
                f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start_epoch}&period2={end_epoch}&interval=1d&events=div&includeAdjustedClose=true"
            )
            .dropna()
            .set_index("Date")
        )

        if dividends is not None and len(dividends) > 1:
            dividends["date"] = pd.to_datetime(dividends.index, format="%Y-%m-%d")
            dividends = (
                dividends.reset_index(drop=True)
                .set_index("date")
                .sort_index()
                .rename(columns={"Dividends": "dividend_amount"})
            )

        return dividends

    @staticmethod
    def _download_splits_yahoo(ticker: str, start_date: str, end_date: str):
        start_epoch = int(start_date.timestamp())
        end_epoch = int(end_date.timestamp())

        splits = (
            pd.read_csv(
                f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start_epoch}&period2={end_epoch}&interval=1d&events=split&includeAdjustedClose=true"
            )
            .dropna()
            .set_index("Date")
        )


        if splits is not None and len(splits) > 1:
            splits["date"] = pd.to_datetime(splits.index, format="%Y-%m-%d")
            splits = (
                splits.reset_index(drop=True)
                .set_index("date")
                .sort_index()
                .rename(columns={"Stock Splits": "split_factor"})
            )

        return splits

    @classmethod
    def yahoo_download_one(cls, signals_ticker, date_from=None, date_until=None):
        if date_from is None:
            date_from = "2000-01-01"
        if date_until is None:
            date_until = datetime.today().strftime("%Y-%m-%d")

        date_from = datetime.strptime(date_from, "%Y-%m-%d")
        date_until = datetime.strptime(date_until, "%Y-%m-%d")

        start_epoch = int(date_from.timestamp())
        end_epoch = int(date_until.timestamp())

        quotes = None

        quotes = (
            pd.read_csv(
                f"https://query1.finance.yahoo.com/v7/finance/download/{signals_ticker}?period1={start_epoch}&period2={end_epoch}&interval=1d&events=history&includeAdjustedClose=true"
            )
            .dropna()
            .set_index("Date")
        )


        if quotes is not None and len(quotes) > 1:
            quotes["date64"] = pd.to_datetime(quotes.index, format="%Y-%m-%d")
            quotes = quotes.reset_index(drop=True).set_index("date64").sort_index()
            quotes.index.name = "date"
            quotes.columns = [
                "open",
                "high",
                "low",
                "close",
                "adjusted_close",
                "volume",
            ]

            dividends = cls._download_dividends_yahoo(
                signals_ticker, date_from, date_until
            )
            splits = cls._download_splits_yahoo(
                signals_ticker, date_from, date_until
            )

            if dividends is not None and len(dividends) >= 1:
                quotes = quotes.join(dividends, how="left")


            if splits is not None and len(splits) >= 1:
                quotes = quotes.join(splits, how="left")

        return quotes


class EODHDDownloaderOHLCV:
    def __init__(self):
        pass

    @staticmethod
    def load_dividends_eodhd(ticker, api_key, date_from):
        """
        Load the splits data from the EOD Historical Data API.
        """
        url = f"https://eodhistoricaldata.com/api/div/{ticker}?api_token={api_key}&fmt=json&from={date_from}"
        response = requests.get(url)

        if response.status_code == 200:
            if len(response.json()) > 0:
                res = (
                    pd.DataFrame(response.json())
                    .set_index("date")
                    .add_prefix("dividend_")
                )
                res.index = pd.to_datetime(res.index, format="%Y-%m-%d")
                res.rename(columns={"dividend_value": "dividend_amount"}, inplace=True)
                # keep only the dividend amount and date
                res = res[["dividend_amount"]]
                return res
        else:
            return None

    @staticmethod
    def load_splits_eodhd(ticker, api_key, date_from):
        """
        Load the splits data from the EOD Historical Data API.
        """
        url = f"https://eodhistoricaldata.com/api/splits/{ticker}?api_token={api_key}&fmt=json&from={date_from}"
        response = requests.get(url)

        if response.status_code == 200:
            if len(response.json()) > 0:
                df = (
                    pd.DataFrame(response.json())
                    .set_index("date")
                    .rename(columns={"split": "split_ratio"})
                )
                # parse the split ratio from string to float: '2.000000/1.000000' -> 2.0
                df["split_ratio"] = df["split_ratio"].apply(
                    lambda x: float(x.split("/")[0])
                )
                df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
                return df
        else:
            return None

    @classmethod
    def eodhd_download_one(
        cls, signals_ticker, api_key, date_from=None, date_until=None
    ):
        if date_from is None:
            start_date = "2000-01-01"
        else:
            start_date = date_from

        quotes = None

        r = requests.get(
            f"https://eodhistoricaldata.com/api/eod/{signals_ticker}?from={start_date}&fmt=json&api_token={api_key}"
        )

        if r.status_code == requests.codes.ok:
            if len(r.json()) > 0:
                quotes = pd.DataFrame(r.json()).set_index("date")
                quotes["date64"] = pd.to_datetime(quotes.index, format="%Y-%m-%d")
                quotes = quotes.reset_index(drop=True).set_index("date64").sort_index()
                quotes.index.name = "date"
                quotes.columns = [
                    "open",
                    "high",
                    "low",
                    "close",
                    "adjusted_close",
                    "volume",
                ]

            dividends = cls.load_dividends_eodhd(signals_ticker, api_key, date_from)
            splits = cls.load_splits_eodhd(signals_ticker, api_key, date_from)

            if dividends is not None:
                quotes: pd.DataFrame = quotes.join(dividends, how="left", on="date")

            if splits is not None:
                quotes: pd.DataFrame = quotes.join(splits, how="left", on="date")

        return quotes


class StockDataDownloader:
    def __init__(self, max_workers=2, eodhd_apikey: str = ""):
        self.max_workers = max_workers
        self.eodhd_apikey = eodhd_apikey

    def download_one(self, bloomberg_ticker, map, eodhd_api_key=None, date_from=None):
        yahoo_ticker = map.loc[bloomberg_ticker, "yahoo"]
        signals_ticker = map.loc[bloomberg_ticker, "signals_ticker"]
        data_provider = map.loc[bloomberg_ticker, "data_provider"]

        if self.eodhd_apikey is None or len(self.eodhd_apikey) == 0:
            data_provider = "yahoo"

        if pd.isnull(signals_ticker):
            return bloomberg_ticker, None

        quotes = None
        for _ in range(3):
            try:
                if data_provider == "eodhd":
                    quotes = EODHDDownloaderOHLCV.eodhd_download_one(
                        signals_ticker, eodhd_api_key, date_from=date_from
                    )
                elif data_provider == "yahoo":
                    quotes = YahooDownloaderOHLCV.yahoo_download_one(
                        signals_ticker=signals_ticker, date_from=date_from
                    )

                if quotes is not None:
                    quotes["data_provider"] = data_provider

                break

            except Exception as ex:
                # logger.exception(ex)
                time.sleep(5)

        return bloomberg_ticker, quotes

    def download_all(self, ticker_map, date_from=None):
        tickers = pd.Series(ticker_map.index).sample(frac=1).unique().tolist()
        print(f"download_all, tickers:{len(tickers)}")

        all_quotes = []
        eodhd_api_key = self.eodhd_apikey

        with futures.ThreadPoolExecutor(self.max_workers) as executor:
            _futures = []
            for ticker in tqdm(tickers):
                _futures.append(
                    executor.submit(
                        self.download_one,
                        bloomberg_ticker=ticker,
                        map=ticker_map,
                        eodhd_api_key=eodhd_api_key,
                        date_from=date_from,
                    )
                )

            print(f"download_all, futures:{len(_futures)}")
            for future in tqdm(futures.as_completed(_futures), total=len(tickers)):
                bloomberg_ticker, quotes = future.result()
                if quotes is not None:
                    quotes["bloomberg_ticker"] = bloomberg_ticker
                    all_quotes.append(quotes)

        return all_quotes


def remove_wrong_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["open"] > 0]
    df = df[df["high"] > 0]
    df = df[df["low"] > 0]
    df = df[df["close"] > 0]
    df = df[df["adjusted_close"] > 0]
    df = df[df["volume"] > 0]
    return df


def re_adjust_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    ratio = df["close"] / df["adjusted_close"]
    df["open"] = df["open"] / ratio
    df["high"] = df["high"] / ratio
    df["low"] = df["low"] / ratio
    df["close"] = df["close"] / ratio
    return df


def update_daily_data(data_dir: str, daily_data_dir: str, EODHD_API_KEY: str = None):
    # read the latest date
    latest_date = get_latest_date(daily_data_dir)

    today_date = datetime.today().strftime("%Y-%m-%d")
    print(f"Today date: {today_date}")
    if today_date == latest_date:
        print("Already up-to-date")
        return
    print(f"Latest date: {latest_date}")

    # download data from the latest date
    downloader = StockDataDownloader(
        max_workers=multiprocessing.cpu_count() - 1, eodhd_apikey=EODHD_API_KEY
    )

    ticker_map_fname = f"{data_dir}/eodhd-map.csv"
    if not os.path.exists(ticker_map_fname):
        print(f"Missing ticker map file: {ticker_map_fname}")
        url = "https://raw.githubusercontent.com/parmarsuraj99/dsignals/main/db/eodhd-map.csv"
        r = requests.get(url)
        if r.status_code == requests.codes.ok:
            with open(ticker_map_fname, "wb") as f:
                f.write(r.content)

    ticker_map = pd.read_csv(f"{data_dir}/eodhd-map.csv").set_index("bloomberg_ticker")

    all_quotes = downloader.download_all(ticker_map, date_from=latest_date)

    # save all quotes
    all_quotes = pd.concat(all_quotes)
    all_quotes = remove_wrong_rows(all_quotes)
    all_quotes = re_adjust_ohlc(all_quotes)

    print(all_quotes.index)
    print(all_quotes.columns)

    print(all_quotes.shape)
    print(all_quotes.head())

    save_in_folders(all_quotes, daily_data_dir)

    # if os.path.exists(f"{data_dir}/all_quotes.parquet"):
    #     _prev_quotes = pd.read_pickle(f"{data_dir}/all_quotes.parquet")
    #     # concat the new quotes with the old ones
    #     all_quotes = pd.concat([_prev_quotes, all_quotes], axis=0)

    # try:
    #     all_quotes.to_parquet(f"{data_dir}/all_quotes.parquet")
    # except Exception as ex:
    #     print(f"Failed to save all quotes: {ex}")
    gc.collect()


if __name__ == "__main__":
    dir_config = Directories()
    update_daily_data(dir_config.DATA_DIR, dir_config.DAILY_DATA_DIR)
    pass
