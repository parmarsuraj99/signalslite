from datetime import datetime
import logging
import pandas as pd
from .base import DownloadFundamentals, DownloadOHLCV


class YahooFinanceOHLCV(DownloadOHLCV):
    DATA_PROVIDER = "yahoo"

    def __init__(self):
        super().__init__()

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
    def download_one_ticker(cls, signals_ticker, date_from=None, date_until=None):
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
            splits = cls._download_splits_yahoo(signals_ticker, date_from, date_until)

            if dividends is not None and len(dividends) >= 1:
                quotes = quotes.join(dividends, how="left")

            if splits is not None and len(splits) >= 1:
                quotes = quotes.join(splits, how="left")

        # logging.info(f"Downloaded {signals_ticker} from Yahoo Finance.")
        # pring dividents adn splits count
        # logging.info(f"Dividends: {dividends.dropna(), splits.dropna()}")

        return quotes


class YahooFinanceFundamentals(DownloadFundamentals):
    DATA_PROVIDER = "yahoo"

    def __init__(self):
        super().__init__()

    def download_one_ticker(self, signals_ticker):
        raise NotImplementedError

    def download_all_tickers(self, tickers):
        raise NotImplementedError


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # print(YahooFinanceOHLCV.download_one_ticker("AAPL"))

    # example with class
    downloader = YahooFinanceOHLCV()
    df = downloader.download_one_ticker("AAPL")
    print(df)
