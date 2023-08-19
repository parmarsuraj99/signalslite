import numpy as np
import pandas as pd
import os
from pathlib import Path
import gc
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

from typing import Union, List, Tuple, Optional, Dict, Any

import itertools

from signalslite.data_utils import (
    load_recent_data_from_file,
    save_daily_data,
    save_in_folders,
    get_latest_date,
    read_available_dates,
)
from signalslite.constants import Directories

try:
    import cudf
    import numba
    from numba import cuda
except ImportError:
    print("cudf not found, using pandas")


def get_combination_ratio(df, feature_prefix: str):
    # fetch feature_name and its combination ratio

    feature_cols = [f for f in df.columns if feature_prefix in f]
    feature_pairs = itertools.combinations(feature_cols, 2)

    _feature_list = []
    for _f1, _f2 in feature_pairs:
        _feature_type = _f1.split("_")[2]  # sma, ema, macd, rsi, etc.
        _f1_window_size = _f1.split("_")[-1]  # window size 1
        _f2_window_size = _f2.split("_")[-1]  # window size 2

        _res = 1 - (df[_f2] / df[_f1])
        _res.name = (
            f"feature_2_ratio_{_feature_type}_{_f1_window_size}_{_f2_window_size}"
        )

        _feature_list.append(_res)

        gc.collect()

    # ratio with close price for each feature
    for _f in feature_cols:
        _feature_type = _f.split("_")[2]
        _f_window_size = _f.split("_")[-1]

        _res = 1 - (df["close"] / df[_f])
        _res.name = f"feature_2_ratio_{_feature_type}_{_f_window_size}_close"

        _feature_list.append(_res)

        gc.collect()

    _cated_res = pd.concat(_feature_list, axis=1).astype("float32")

    del _feature_list
    gc.collect()

    return _cated_res


def calculate_all_secondary_features(
    df, feature_prefixes: List[str]
):
    _all_features = []
    for feature_prefix in feature_prefixes:
        _features = [f for f in df.columns if feature_prefix in f]
        _res = get_combination_ratio(df.loc[:, _features + ["close"]], feature_prefix)
        _all_features.append(_res)
        gc.collect()

    _all_features = pd.concat(_all_features, axis=1).astype("float32")
    return _all_features


def update_secondary_features(dir_config, feature_prefixes=None):
    if feature_prefixes is None:
        feature_prefixes = [
            "feature_1_sma",
            "feature_1_ema",
            "feature_1_rsi",
        ]
    start_index = 0
    primary_data_dates = read_available_dates(dir_config.DAILY_PRIMARY_FEATURES_DIR)
    print(f"primary_dates: {len(primary_data_dates)}")
    dates = primary_data_dates

    if os.path.exists(dir_config.DAILY_SECONDARY_FEATURES_DIR):
        secondary_data_dates = read_available_dates(
            dir_config.DAILY_SECONDARY_FEATURES_DIR
        )
        print(f"secondary_features_dates: {len(secondary_data_dates)}")

        n_days_to_load = len(primary_data_dates) - len(secondary_data_dates) + 10

        print(f"n_days_to_load: {n_days_to_load}")

        dates = primary_data_dates
        start_index = max(
            0, len(secondary_data_dates) - 1000
        )  # to avoid getting negative dates

    # iterate over all dates in chunks of 200
    for i in tqdm(range(start_index, len(dates), 1000)):
        print(i)
        _df = load_recent_data_from_file(
            dir_config.DAILY_PRIMARY_FEATURES_DIR,
            n_days=1000,
            ascending=True,
            offset=i,
            dtype="float32",
        )

        feat_cols = [f for f in _df if "feature_" in f]

        _res = calculate_all_secondary_features(_df, feature_prefixes).astype("float32")

        # combine primary and secondary featuers
        _res = pd.concat([_df, _res], axis=1)
        _res = _res.replace([np.inf, -np.inf], np.nan)
        _res = _res.dropna(axis=0)

        assert (
            _res.isna().mean().sort_values(ascending=False).max() < 0.1
        ), "too many NaN values found"
        save_in_folders(_res, dir_config.DAILY_SECONDARY_FEATURES_DIR)

        # del _df, _res
        gc.collect()


if __name__ == "__main__":
    dir_config = Directories()

    feature_prefixes = [
        "feature_1_sma",
        "feature_1_ema",
        "feature_1_rsi",
    ]

    update_secondary_features(dir_config, feature_prefixes)
