import logging
import time

import pandas as pd

from signalslite.downloaders.eodhd import EODHDOHLCV, EOODHDFundamentals
from signalslite.downloaders.yahoo_finance import (
    YahooFinanceFundamentals,
    YahooFinanceOHLCV,
)


def map_ticker_to_provider(ticker_map: pd.DataFrame):
    """
    Map the ticker symbols from one exchange to another.

    Parameters
    ----------
    ticker_map: pd.DataFrame
        A DataFrame containing the mapping of the ticker symbols from one exchange to another.

    Returns
    -------
    A dictionary containing the mapping of the ticker symbols from one exchange to another.
    """

    unique_providers = ticker_map["data_provider"].dropna().unique().tolist()
    unique_providers = [
        u for u in unique_providers if len(u) > 2 and u.lower() != "ignore"
    ]
    logging.info(f"Unique providers: {unique_providers}")

    provider_ticker_map = {}
    for provider in unique_providers:
        slice = ticker_map[ticker_map["data_provider"] == provider]

        logging.info(f"Ticker: {provider}, Slice: {len(slice)}")
        bbg_to_provider = slice["signals_ticker"].dropna().to_dict()

        provider_ticker_map[provider] = bbg_to_provider

    tickers = []
    data_providers = []
    bbg_tickers = []
    for data_provider, provider_ticker_map in provider_ticker_map.items():
        tickers.extend(provider_ticker_map.values())
        bbg_tickers.extend(provider_ticker_map.keys())
        data_providers.extend([data_provider] * len(provider_ticker_map.values()))

    tickers_df = (
        pd.DataFrame(
            {
                "ticker": tickers,
                "data_provider": data_providers,
                "bbg_ticker": bbg_tickers,
            }
        )
        .drop_duplicates()
        .sample(frac=1)
        # .head(10)
    )

    return tickers_df


class OHLCVDownloader:
    def __init__(self, eodhd_api_key: str, tickermap: dict = None):
        """
        Initialize the OHLCVDownloader class with the provided downloader.

        Parmeters
        ---------
        eodhd_api_key: str
            The API key for EOD Historical Data API.
        tickermap: dict
            A dictionary mapping the ticker symbols from one exchange to another.
        """
        self.eodhd_api_key = eodhd_api_key
        self.eodhd_downloader = EODHDOHLCV(eodhd_api_key)
        self.yahoo_downloader = YahooFinanceOHLCV()
        self.tickermap = tickermap

    def download_one(
        self, provider_ticker: str, data_provider: str, date_from=None, date_until=None
    ):
        """
        Download the OHLCV data for a single ticker.

        Parameters
        ----------
        ticker: str
            The ticker symbol of the stock.
        date_from: str
            The start date for the data. Format: YYYY-MM-DD
        date_until: str
            The end date for the data. Format: YYYY-MM-DD

        Returns
        -------
        A DataFrame containing the OHLCV data.
        """

        # logging.info(f"Downloading {provider_ticker} from {data_provider}.")

        quotes = None
        for _ in range(3):
            try:
                if data_provider == "eodhd":
                    quotes = self.eodhd_downloader.download_one_ticker(
                        signals_ticker=provider_ticker,
                        date_from=date_from,
                        date_until=date_until,
                    )
                elif data_provider == "yahoo":
                    quotes = self.yahoo_downloader.download_one_ticker(
                        signals_ticker=provider_ticker,
                        date_from=date_from,
                        date_until=date_until,
                    )

                if quotes is not None:
                    quotes["data_provider"] = data_provider

                break

            except Exception as ex:
                logging.error(
                    f"Error downloading {provider_ticker} from {data_provider}: {ex}"
                )
                time.sleep(5)

        return provider_ticker, quotes

    def download_all(self, date_from=None, date_until=None):
        """
        Download the OHLCV data for a list of tickers.

        Parameters
        ----------
        tickers: list
            A list of ticker symbols.
        date_from: str
            The start date for the data. Format: YYYY-MM-DD
        date_until: str
            The end date for the data. Format: YYYY-MM-DD

        Returns
        -------
        A dictionary containing the OHLCV data for each ticker.
        """

        tickers_df = map_ticker_to_provider(self.tickermap)

        # use futures to download the data in parallel; sample from provider maps to avoid hitting the API limits
        import concurrent.futures

        from tqdm import tqdm

        # download the data in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {
                executor.submit(
                    self.download_one,
                    provider_ticker=ticker,
                    data_provider=data_provider,
                    date_from=date_from,
                    date_until=date_until,
                ): bbg_ticker
                for ticker, data_provider, bbg_ticker in zip(
                    tickers_df["ticker"],
                    tickers_df["data_provider"],
                    tickers_df["bbg_ticker"],
                )
            }

            logging.info(f"Number of tickers: {len(future_to_ticker)}")

            results = []
            pbar = tqdm(total=len(future_to_ticker))
            for future in tqdm(
                concurrent.futures.as_completed(future_to_ticker),
                total=len(future_to_ticker),
            ):
                bbg_ticker = future_to_ticker[future]
                try:
                    provider_ticker, quotes = future.result()
                    if quotes is not None:
                        quotes["bloomberg_ticker"] = bbg_ticker
                        results.append(quotes)
                        pbar.update(1)
                except Exception as exc:
                    logging.error(f"Error downloading {provider_ticker} from: {exc}")

            pbar.close()
            # return as a dataframe with bloomberg ticker
            logging.info(f"Downloaded {len(results)} quotes.")

            return results


class StockFundamentalDownloader:
    def __init__(self, eodhd_api_key: str, tickermap: dict = None):
        self.eodhd_downloader = EOODHDFundamentals(eodhd_api_key)
        self.tickermap = tickermap

    def download_one(self, provider_ticker: str, data_provider: str):
        for _ in range(3):
            try:
                if data_provider == "eodhd":
                    data = self.eodhd_downloader.download_one_ticker(provider_ticker)
                    return provider_ticker, data
            except Exception as ex:
                logging.error(
                    f"Error downloading {provider_ticker} from {data_provider}: {ex}"
                )
                time.sleep(5)

        return provider_ticker, None

    def download_all(self, mappig: pd.DataFrame):
        tickers_df = map_ticker_to_provider(mappig)

        # use futures to download the data in parallel; sample from provider maps to avoid hitting the API limits
        import concurrent.futures

        from tqdm import tqdm

        # download the data in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {
                executor.submit(
                    self.download_one,
                    provider_ticker=ticker,
                    data_provider=data_provider,
                ): bbg_ticker
                for ticker, data_provider, bbg_ticker in zip(
                    tickers_df["ticker"],
                    tickers_df["data_provider"],
                    tickers_df["bbg_ticker"],
                )
            }

            logging.info(f"Number of tickers: {len(future_to_ticker)}")

            results = {}
            pbar = tqdm(total=len(future_to_ticker))
            for future in tqdm(
                concurrent.futures.as_completed(future_to_ticker),
                total=len(future_to_ticker),
            ):
                bbg_ticker = future_to_ticker[future]
                try:
                    provider_ticker, quotes = future.result()
                    if quotes is not None:
                        results[bbg_ticker] = quotes
                        pbar.update(1)
                except Exception as exc:
                    logging.error(f"Error downloading {provider_ticker} from: {exc}")

            pbar.close()
            # return as a dataframe with bloomberg ticker
            logging.info(f"Downloaded {len(results)} quotes.")

            return results


if __name__ == "__main__":
    import logging
    import os

    logging.basicConfig(level=logging.INFO)

    mappig = pd.read_csv("data/eodhd-map.csv").set_index("bloomberg_ticker")

    EODHD_API_KEY = os.environ.get("EODHD_API_KEY")

    # downloader = OHLCVDownloader(EODHD_API_KEY, mappig)
    # all_quotes = downloader.download_all()
    # pd.concat(all_quotes).to_csv("data/eodhd.csv")
    # logging.info(f"Downloaded {len(all_quotes)} from EOD Historical Data.")

    fundamental_downloader = StockFundamentalDownloader(EODHD_API_KEY, mappig)
    fundamental_data = fundamental_downloader.download_all(mappig)

    pd.to_pickle(fundamental_data, f"data/eodhd_fundamentals.pkl")
