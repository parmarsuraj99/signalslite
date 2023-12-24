from abc import ABC
import pandas as pd
import os
from pathlib import Path

class StorageBase(ABC):
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)

    def list_root_dir(self):
        return os.listdir(self.root_dir)
        
class StorageReader(StorageBase):
    """
    Need something like this to read the data from the local storage.
    which can be extended to read from the cloud storage as well.
    a functionalty to read subdir and files from the subdir.
    For ohlcv and fundamentals, we can read the data from the same subdir.
    """
    def __init__(self, root_dir: str):
        super().__init__(root_dir)
        
    def read(self, ticker: str, data_type: str, data_provider: str):
        raise NotImplementedError
    

class StorageWriter(StorageBase):
    """
    Need something like this to write the data to the local storage.
    which can be extended to write to the cloud storage as well.
    a functionalty to write subdir and files to the subdir.
    For ohlcv and fundamentals, we can write the data to the same subdir.
    """
    def __init__(self, root_dir: str):
        super().__init__(root_dir)

    def write(self, ticker: str, data_type: str, data_provider: str, data: pd.DataFrame):
        raise NotImplementedError
    