from abc import ABC, abstractmethod


class DownloaderBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def download_one_ticker(self, ticker, start, end):
        raise NotImplementedError

    @abstractmethod
    def download_all_tickers(self, tickers, start, end):
        # in case if we want to download all tickers from a specific exchange; 
        # using a ticker map now
        raise NotImplementedError


class DownloadOHLCV(DownloaderBase):
    def __init__(self):
        super().__init__()

    def download_one_ticker(self, ticker, start, end):
        raise NotImplementedError

    def download_all_tickers(self, tickers, start, end):
        raise NotImplementedError


class DownloadFundamentals(DownloaderBase):
    def __init__(self):
        super().__init__()

    def download_one_ticker(self, ticker, start, end):
        raise NotImplementedError

    def download_all_tickers(self, tickers, start, end):
        raise NotImplementedError
