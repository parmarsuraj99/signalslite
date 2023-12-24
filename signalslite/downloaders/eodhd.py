import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
from base import DownloadFundamentals, DownloadOHLCV


class EODHDOHLCV(DownloadOHLCV):
    DATA_PROVIDER = "eodhd"
    """
    A class used to download OHLCV data from EOD Historical Data API.
    """

    def __init__(self, eodhd_api_key: str):
        """
        Initialize the EODHDOHLCV class with the provided API key.

        """
        super().__init__()
        self.eodhd_api_key = eodhd_api_key

    @staticmethod
    def load_dividends_eodhd(
        ticker: str, api_key: str, date_from: str
    ) -> Optional[pd.DataFrame]:
        """
        Load the dividends data from the EOD Historical Data API.

        """
        url = f"https://eodhd.com/api/div/{ticker}?from={date_from}&api_token={api_key}&fmt=json"
        # logging.info(f"Requesting dividends: {url}")
        response = requests.get(url)
        # logging.info(f"Response for dividends:`{response.status_code}`")

        if int(response.status_code) == 200:
            data = response.json()
            # logging.info(f"Response for dividends: {data}")
            if data and len(data) > 0:
                res = pd.DataFrame(data).set_index("date").add_prefix("dividend_")
                # logging.info(f"Response for dividends: {res}")

                res.index = pd.to_datetime(res.index, format="%Y-%m-%d")
                res.rename(columns={"dividend_value": "dividend_amount"}, inplace=True)
                return res
        else:
            return None

    @staticmethod
    def load_splits_eodhd(
        ticker: str, api_key: str, date_from: str
    ) -> Optional[pd.DataFrame]:
        """
        Load the splits data from the EOD Historical Data API.

        """
        url = f"https://eodhd.com/api/splits/{ticker}?from={date_from}&api_token={api_key}&fmt=json"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            if data:
                df = (
                    pd.DataFrame(data)
                    .set_index("date")
                    .rename(columns={"split": "split_ratio"})
                )
                # parse the split ratio from string to float: '2.000000/1.000000' -> 2.0
                df["split_ratio"] = df["split_ratio"].apply(
                    lambda x: float(x.split("/")[0]) if "/" in x else float(x)
                )
                df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
                return df
        else:
            return None

    def download_one_ticker(
        self,
        signals_ticker: str,
        date_from: Optional[str] = None,
        date_until: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Download the OHLCV data for one ticker from the EOD Historical Data API.
        """
        if date_from is None:
            start_date = "2000-01-01"

        if date_until is None:
            date_until = datetime.today().strftime("%Y-%m-%d")

        quotes = None
        api_key = self.eodhd_api_key

        r = requests.get(
            f"https://eodhd.com/api/eod/{signals_ticker}?from={start_date}&api_token={api_key}&fmt=json"
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

            dividends = self.load_dividends_eodhd(signals_ticker, api_key, start_date)
            splits = self.load_splits_eodhd(signals_ticker, api_key, start_date)

            if dividends is not None:
                quotes: pd.DataFrame = quotes.join(dividends, how="left", on="date")

            if splits is not None:
                quotes: pd.DataFrame = quotes.join(splits, how="left", on="date")

        return quotes


class EOODHDFundamentals(DownloadFundamentals):
    def __init__(self, eodhd_api_key: str):
        super().__init__()
        self.eodhd_api_key = eodhd_api_key

    def download_one_ticker(self, signals_ticker: str):
        url = f"https://eodhd.com/api/fundamentals/{signals_ticker}?api_token={self.eodhd_api_key}&fmt=json"
        response = requests.get(url)

        if int(response.status_code) == 200:
            data = response.json()

            # data will be very nested, so we simply return the whole thing
            return data

    def download_all_tickers(self, tickers: list):
        raise NotImplementedError


if __name__ == "__main__":
    import logging
    import os

    # logging.basicConfig(level=# logging.INFO)

    EODHD_API_KEY = os.environ.get("EODHD_API_KEY")
    downloader = EODHDOHLCV(EODHD_API_KEY)
    df = downloader.download_one_ticker("AAPL.US")

    fundamental_downloader = EOODHDFundamentals(EODHD_API_KEY)
    fundamental_data = fundamental_downloader.download_one_ticker("AAPL.US")

    pd.to_pickle(df, f"data/eodhd_aapl.pkl")

    # logging.info(f"Downloaded data for AAPL: {df}")
