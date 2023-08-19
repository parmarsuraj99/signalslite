import pandas as pd
import numpy as np
import os
from pathlib import Path
import gc
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

from signalslite.data_utils import (
    load_recent_data_from_file,
    save_in_folders,
    get_latest_date,
    read_available_dates,
)
from signalslite.constants import Directories

from signalslite.technical_features import (
    simple_moving_average,
    exponential_moving_average,
    bollinger_bands,
    rsi,
    macd,
    average_true_range,
)

try:
    import cudf
    import numba
    from numba import cuda

    USE_CUDF = True
except ImportError:
    USE_CUDF = False
    print("cudf not found, using pandas")


def load_recent_data(DAILY_DATA_DIR, n_days):
    recent_data = (
        load_recent_data_from_file(DAILY_DATA_DIR, n_days=n_days, ascending=False)
        .reset_index()
        .sort_values(by=["bloomberg_ticker", "date"])
    )

    cols = recent_data.select_dtypes(include=["float32", "int64"]).columns

    recent_data[cols] = recent_data[cols].astype("float16")

    # filter out tickers with less than 100 days of data
    recent_data = recent_data.groupby("bloomberg_ticker").filter(lambda x: len(x) > 100)
    recent_data = recent_data.groupby("date").filter(lambda x: len(x) > 500)
    gc.collect()

    recent_data.info()

    return recent_data


def compute_features(df, function_to_window, use_cudf:bool):
    features = []
    for func, windows in function_to_window.items():
        for window in windows:
            # pass windows as a tuple if the function takes more than one window
            if isinstance(window, tuple):
                _feat = func(df, *window)
            else:
                _feat = func(df, window)

            if isinstance(_feat, tuple):
                features.extend(_feat)
            else:
                features.append(_feat)

    # print type of features
    if use_cudf:
        print("features type: cudf")
        cated = cudf.concat(features, axis=1).astype("float32").add_prefix("feature_1_")
    else:
        print("features type: pandas")
        cated = pd.concat(features, axis=1).astype("float32").add_prefix("feature_1_")
    return cated


def generate_features(recent_data, function_to_window, use_cudf:bool):
    tickers_list = recent_data["bloomberg_ticker"].unique().tolist()

    # iterate over ticker chunks in 500
    res = []
    for i in tqdm(range(0, len(tickers_list), 1000)):
        tickers = tickers_list[i : i + 1000]
        # print(tickers)
        tickers_data = recent_data[recent_data["bloomberg_ticker"].isin(tickers)]
        if use_cudf:
            _df_gpu = cudf.from_pandas(tickers_data)
        else:
            _df_gpu = tickers_data

        _res = compute_features(_df_gpu, function_to_window, use_cudf=use_cudf)
        if use_cudf:
            _res = _res.to_pandas().astype("float16")
            _res["date"] = _df_gpu["date"].to_pandas()
            _res["bloomberg_ticker"] = _df_gpu["bloomberg_ticker"].to_pandas()
            _res["close"] = _df_gpu["close"].to_pandas()
            _res["volume"] = _df_gpu["volume"].to_pandas()
            _res["open"] = _df_gpu["open"].to_pandas()
            _res["high"] = _df_gpu["high"].to_pandas()
            _res["low"] = _df_gpu["low"].to_pandas()
        else:
            _res["date"] = _df_gpu["date"]
            _res["bloomberg_ticker"] = _df_gpu["bloomberg_ticker"]
            _res["close"] = _df_gpu["close"]
            _res["volume"] = _df_gpu["volume"]
            _res["open"] = _df_gpu["open"]
            _res["high"] = _df_gpu["high"]
            _res["low"] = _df_gpu["low"]

        res.append(_res)

        del _df_gpu, _res

        gc.collect()
        res = pd.concat(res, axis=0)
        res = res.dropna(axis=0)

        # convert float 16 to float 32 in a loop
        for col in res.columns:
            if res[col].dtype == "float16":
                res[col] = res[col].astype("float32")
            gc.collect()

        gc.collect()

        return res


def save_features(res, DAILY_PRIMARY_FEATURES_DIR):
    # loop over all unique dates in chunks of 100; save each chunk to a separate file
    res["date_str"] = res["date"].dt.strftime("%Y-%m-%d")
    dates = res["date_str"].unique()

    for i in tqdm(range(0, len(dates), 100)):
        # use save_in_folders function to save each chunk to a separate folder
        _tmp = res[res["date_str"].isin(dates[i : i + 100])]
        save_in_folders(_tmp, DAILY_PRIMARY_FEATURES_DIR)

        del _tmp
        gc.collect()


def generate_primary_features(dir_config):

    try:
        import cudf
        import numba
        from numba import cuda

        USE_CUDF = True
    except ImportError:
        USE_CUDF = False
        print("cudf not found, using pandas")

    n_days_to_load = -1

    # if some of primary features in days are there then take last 1000 days in adjusted data: 1000
    # else take all days in adjusted data: -1
    if os.path.exists(dir_config.DAILY_PRIMARY_FEATURES_DIR):
        raw_data_dates = read_available_dates(dir_config.DAILY_DATA_DIR)
        print(f"raw_data_dates: {len(raw_data_dates)}")
        primary_features_dates = read_available_dates(
            dir_config.DAILY_PRIMARY_FEATURES_DIR
        )

        # difference in days between last date in raw data and last date in primary features
        last_date_raw = np.sort(raw_data_dates)[-1]
        last_date_primary = np.sort(primary_features_dates)[-1]
        diff = (pd.to_datetime(last_date_raw) - pd.to_datetime(last_date_primary)).days
        print(diff)
        print(f"primary_features_dates: {len(primary_features_dates)}")

        n_days_to_load = diff + 1000

    print(f"n_days_to_load: {n_days_to_load}")

    function_to_window: dict = {
        simple_moving_average: [5, 10, 20, 50, 100, 200],
        exponential_moving_average: [5, 10, 20, 50, 100, 200],
        bollinger_bands: [5, 10, 20, 50, 100, 200],
        rsi: [5, 10, 20, 50, 100, 200],
        average_true_range: [5, 10, 20, 50, 100, 200],
        macd: [(12, 26), (20, 50)],
    }

    # load recent data
    recent_data = load_recent_data(dir_config.DAILY_DATA_DIR, n_days_to_load)
    features = generate_features(recent_data, function_to_window, USE_CUDF)
    save_features(features, dir_config.DAILY_PRIMARY_FEATURES_DIR)


if __name__ == "__main__":
    dir_config = Directories()
    generate_primary_features(dir_config)
