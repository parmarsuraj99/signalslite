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

from signalslite.constants import (
    DATA_DIR,
    DAILY_DATA_DIR,
)

print(DATA_DIR, DAILY_DATA_DIR)

