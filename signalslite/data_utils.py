import os
from pathlib import Path
import pandas as pd
import gc
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from functools import lru_cache


def read_available_dates(DAILY_DATA_DIR):
    dates = []
    for root, dirs, files in os.walk(DAILY_DATA_DIR):
        for file in files:
            if file.endswith(".parquet"):
                dates.append(file[:-8])
    dates = sorted(dates)
    return dates


@lru_cache(maxsize=256)
def load_recent_data_from_file(
    DAILY_DATA_DIR, n_days=1, ascending=False, offset=0, dtype: str = "float32"
):
    assert dtype in ["float32", "float16"]
    print(DAILY_DATA_DIR)

    dates = []
    for root, dirs, files in os.walk(DAILY_DATA_DIR):
        for file in files:
            if file.endswith(".parquet"):
                dates.append(file[:-8])
    _to_reverse = ascending == False
    dates = sorted(dates, reverse=_to_reverse)

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
            float16_cols = _tmp.select_dtypes(include=["float32"]).columns
            _tmp[float16_cols] = _tmp[float16_cols].astype("float16")

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
        float16_cols = _tmp.select_dtypes(include=["float32"]).columns
        _tmp[float16_cols] = _tmp[float16_cols].astype("float16")

    return _tmp


def save_daily_data(df, date, level_path):
    year = date[:4]
    month = date[5:7]
    day = date[8:10]
    Path(os.path.join(level_path, year, month)).mkdir(parents=True, exist_ok=True)
    f_name = os.path.join(level_path, year, month, f"{date}.parquet")
    # print(f_name)
    if not os.path.exists(f_name):
        # print(f"Saving {date} data to {f_name}")
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

    # save data for each date in a separate file with parquet format
    res = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(save_daily_data)(group, date, level_path)
        for date, group in tqdm(df.groupby("date_str"))
    )
    del res

    gc.collect()


def get_latest_date(DAILY_DATA_DIR: str):
    dates = []
    for root, dirs, files in os.walk(DAILY_DATA_DIR):
        for file in files:
            if file.endswith(".parquet"):
                dates.append(file[:-8])
    return max(dates) if len(dates) > 0 else "2000-01-01"
