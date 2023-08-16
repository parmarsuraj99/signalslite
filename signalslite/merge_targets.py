import pandas as pd
import os
from pathlib import Path
import gc
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

import cudf
import numba
from numba import cuda
import numpy as np
import numerapi

# parallelize the process on all columns using joblib
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

from signalslite.data_utils import (
    load_recent_data_from_file,
    save_daily_data,
    save_in_folders,
    get_latest_date,
    read_available_dates,
)
from signalslite.constants import Directories


def update_historical_file(dir_config):
    napi = numerapi.SignalsAPI(verbosity="info")
    # get the latest date
    pd.read_csv(napi.HISTORICAL_DATA_URL).to_csv(
        dir_config.DATA_DIR / "numerai_signals_historical.csv", index=False
    )


def merge_data(historical_df: pd.DataFrame, dir_config: Directories):
    dates = read_available_dates(dir_config.DAILY_SCALED_FEATURES_DIR)

    merged_df = []

    for i in tqdm(range(0, len(dates), 200)):
        _tmp = load_recent_data_from_file(
            dir_config.DAILY_SCALED_FEATURES_DIR, n_days=200, ascending=True, offset=i
        )
        _tmp = _tmp.reset_index(drop=True)
        _tmp = _tmp.sort_values(["date", "bloomberg_ticker"])
        feature_columns = [f for f in _tmp.columns if f.startswith("feature")]

        _historical_dates = historical_df["date"].unique()
        # find common dates
        common_dates = list(
            set(_tmp["date"].unique()).intersection(set(_historical_dates))
        )

        if len(common_dates) == 0:
            continue

        # merge historical_df and _tmp on date and bloomberg_ticker
        _merged = pd.merge(
            historical_df,
            _tmp,
            how="right",
            left_on=["date", "bloomberg_ticker"],
            right_on=["date", "bloomberg_ticker"],
        )
        _merged = _merged.dropna(subset=["target_20d"], axis=0)

        # assert abs(_merged["feature_2_ratio_rsi_50_close"].mean() - 2) < 0.05

        if len(_merged) > 0:
            merged_df.append(_merged)

    merged_df = pd.concat(merged_df)

    return merged_df


def prepare_live(dir_config):
    latest_date = get_latest_date(dir_config.DAILY_SCALED_FEATURES_DIR)
    print("latest date: ", latest_date)
    # load the latest data
    df = load_recent_data_from_file(
        dir_config.DAILY_SCALED_FEATURES_DIR, n_days=1, ascending=False
    )
    df = df.reset_index(drop=True)
    print("df shape: ", df.shape)

    return df


def prepare_merged_data(dir_config):
    update_historical_file(dir_config)
    historical_df = pd.read_csv(dir_config.DATA_DIR / "numerai_signals_historical.csv")
    historical_df["date"] = pd.to_datetime(
        historical_df["friday_date"], format="%Y%m%d"
    )

    merged_df = merge_data(historical_df, dir_config)
    live_df = prepare_live(dir_config)

    merged_df.to_parquet(
        dir_config.DATA_DIR / "merged_data_historical.parquet", index=False
    )
    live_df.to_parquet(dir_config.DATA_DIR / "merged_data_live.parquet", index=False)


if __name__ == "__main__":
    dir_config = Directories()
    dir_config.set_data_dir("data")

    print(dir_config)
    prepare_merged_data(dir_config)
