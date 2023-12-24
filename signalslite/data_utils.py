import os
from pathlib import Path
import pandas as pd
import gc
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from functools import lru_cache

def get_dates_from_files(DAILY_DATA_DIR):
    return [file[:-8] for root, dirs, files in os.walk(DAILY_DATA_DIR) for file in files if file.endswith(".parquet")]

def read_available_dates(DAILY_DATA_DIR):
    return sorted(get_dates_from_files(DAILY_DATA_DIR))

@lru_cache(maxsize=256)
def load_recent_data_from_file(
    DAILY_DATA_DIR, n_days=1, ascending=False, offset=0, dtype: str = "float32"
):
    assert dtype in ["float32", "float16"]
    print(DAILY_DATA_DIR)

    dates = sorted(get_dates_from_files(DAILY_DATA_DIR), reverse=not ascending)

    if n_days < 0:
        _tmp = pd.concat(
            [
                pd.read_parquet(
                    os.path.join(DAILY_DATA_DIR, date[:4], date[5:7], f"{date}.parquet")
                )
                for date in dates
            ]
        )
        if dtype == "float16":
            _tmp = _tmp.astype("float16")
        return _tmp
    
    dates = dates[offset: offset + n_days]

    _tmp = pd.concat(
        [
            pd.read_parquet(
                os.path.join(DAILY_DATA_DIR, date[:4], date[5:7], f"{date}.parquet")
            )
            for date in dates[-n_days:]
        ]
    )
    if dtype == "float16":
        _tmp = _tmp.astype("float16")

    if "open" in _tmp.columns:
        _tmp[[
            "open", "high", "low", "close"
        ]] = _tmp[[
            "open", "high", "low", "close"
        ]].astype(dtype=dtype)

    return _tmp

def save_daily_data(df, date, level_path):
    year, month, day = date.split('-')
    Path(os.path.join(level_path, year, month)).mkdir(parents=True, exist_ok=True)
    f_name = os.path.join(level_path, year, month, f"{date}.parquet")
    if not os.path.exists(f_name):
        df.to_parquet(f_name)

def save_in_folders(df, level_path):
    if "date_str" not in df.columns and "date" not in df.columns:
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df["date_str"] = df.index.strftime("%Y-%m-%d")

    if "date_str" not in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    Path(os.path.join(level_path)).mkdir(parents=True, exist_ok=True)

    Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(save_daily_data)(group, date, level_path)
        for date, group in tqdm(df.groupby("date_str"))
    )
    gc.collect()

def get_latest_date(DAILY_DATA_DIR: str):
    return max(get_dates_from_files(DAILY_DATA_DIR)) if get_dates_from_files(DAILY_DATA_DIR) else "2000-01-01"

