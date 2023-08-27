import pandas as pd
import numpy as np
import os
from pathlib import Path
import gc
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
tqdm.pandas()

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
    chaikin_money_flow,
    average_directional_index,
    commodity_channel_index,
)
from functools import lru_cache

try:
    import cudf
    import numba
    from numba import cuda

    USE_CUDF = True
except ImportError:
    USE_CUDF = False
    print("cudf not found, using pandas")


@lru_cache(maxsize=256)
def load_recent_data(DAILY_DATA_DIR, n_days):
    recent_data = (
        load_recent_data_from_file(DAILY_DATA_DIR, n_days=n_days, ascending=False)
        .reset_index()
        .sort_values(by=["bloomberg_ticker", "date"])
    )

    cols = recent_data.select_dtypes(include=["float32", "int64"]).columns

    recent_data[cols] = recent_data[cols].astype("float16")

    recent_data["volume"] = recent_data["volume"].astype("int")

    # filter out tickers with less than 100 days of data
    recent_data = recent_data.groupby("bloomberg_ticker").filter(lambda x: len(x) > 100)
    recent_data = recent_data.groupby("date").filter(lambda x: len(x) > 400)
    print(recent_data)
    gc.collect()

    recent_data.info()

    return recent_data


def compute_features(df, function_to_window, use_cudf: bool):
    features = []
    for func, windows in function_to_window.items():
        # print(f"computing {func.__name__}")
        for window in windows:
            # pass windows as a tuple if the function takes more than one window
            if isinstance(window, tuple):
                _feat = func(df, *window)
            else:
                _feat = func(df, window)

            if isinstance(_feat, tuple):
                # to float32
                _feat = [feat.astype("float32") for feat in _feat]
                features.extend(_feat)
            else:
                _feat = _feat.astype("float32")
                features.append(_feat)

    # print type of features
    if use_cudf:
        # print("features type: cudf")
        cated = cudf.concat(features, axis=1).astype("float32").add_prefix("feature_1_")
    else:
        # print("features type: pandas")
        cated = pd.concat(features, axis=1).astype("float32").add_prefix("feature_1_")
    return cated


def generate_features(recent_data, function_to_window, use_cudf: bool, dir_config):
    tickers_list = recent_data["bloomberg_ticker"].unique().tolist()
    unique_dates = sorted(recent_data["date_str"].unique().tolist())

    # iterate over ticker chunks in 500
    res = []
    for i in tqdm(range(0, len(tickers_list), 1000)):
        tickers = tickers_list[i : i + 1000]
        # print(tickers)
        tickers_data = recent_data[
            recent_data["bloomberg_ticker"].isin(tickers)
        ].sort_values(by="date")
        if use_cudf:
            _df_gpu = cudf.from_pandas(tickers_data)
        else:
            _df_gpu = tickers_data

        OHLCV_COLS = ["open", "high", "low", "close", "volume"]
        _df_gpu[OHLCV_COLS] = _df_gpu[OHLCV_COLS].astype("float32")

        # _res = compute_features(
        #     _df_gpu[OHLCV_COLS], function_to_window, use_cudf=use_cudf
        # ).copy()

        _res = _df_gpu.groupby("bloomberg_ticker").apply(
            lambda x: compute_features(x[OHLCV_COLS], function_to_window, use_cudf)
        )

        if use_cudf:
            _res = _res.to_pandas().astype("float32")
            _res["date"] = _df_gpu["date"].to_pandas()
            _res["date_str"] = _df_gpu["date_str"].to_pandas()
            _res["bloomberg_ticker"] = _df_gpu["bloomberg_ticker"].to_pandas()
            _res["close"] = _df_gpu["close"].to_pandas()
            _res["volume"] = _df_gpu["volume"].to_pandas()
            _res["open"] = _df_gpu["open"].to_pandas()
            _res["high"] = _df_gpu["high"].to_pandas()
            _res["low"] = _df_gpu["low"].to_pandas()
            # if "split_factor" in _df_gpu.columns:
            #     _res["split_factor"] = _df_gpu["split_factor"].to_pandas()
            #     _res["split_ratio"] = _df_gpu["split_ratio"].to_pandas()
            #     _res["cumulative_split_ratio"] = (
            #         _df_gpu["split_ratio"].cumprod().to_pandas()
            #     )
            # if "dividend_amount" in _df_gpu.columns:
            #     _res["dividend_amount"] = _df_gpu["dividend_amount"].to_pandas()

        else:
            _res["date"] = _df_gpu["date"]
            _res["date_str"] = _df_gpu["date_str"]
            _res["bloomberg_ticker"] = _df_gpu["bloomberg_ticker"]
            _res["close"] = _df_gpu["close"]
            _res["volume"] = _df_gpu["volume"]
            _res["open"] = _df_gpu["open"]
            _res["high"] = _df_gpu["high"]
            _res["low"] = _df_gpu["low"]
            # if "split_factor" in _df_gpu.columns:
            #     _res["split_factor"] = _df_gpu["split_factor"]
            #     _res["split_ratio"] = _df_gpu["split_ratio"]
            #     _res["cumulative_split_ratio"] = _df_gpu["split_ratio"].cumprod()
            # if "dividend_amount" in _df_gpu.columns:
            #     _res["dividend_amount"] = _df_gpu["dividend_amount"]

        res.append(_res)
        print(_res[["date", "close", "feature_1_sma_10", "feature_1_sma_20", "bloomberg_ticker"]].tail(10))
        gc.collect()

        del _df_gpu, _res
        gc.collect()

    # res is an array of tickers with all dates in each ticker
    # iterate over the list and extract dates in chunk of 500
    # concat and save

    gc.collect()

    for ix in range(0, len(unique_dates), 200):
        _dates = unique_dates[ix : ix + 200]
        print(_dates)
        _tmp = []
        for _df in res:
            print("res", _df.shape)
            np.intersect1d(_df["date_str"].unique(), _dates)
            _tmp.append(_df[_df["date_str"].isin(_dates)])
        features = pd.concat(_tmp, axis=0)

        print("cated", features.shape)

        print("before na", features.groupby("date").apply(len).max())
        
        # print inf
        print("inf", features.isin([np.inf, -np.inf]).mean().sort_values())

        print("nan", features.isna().mean().sort_values())
        cols_to_consider_for_dropna = [f for f in features.columns if "split" not in f]
        cols_to_consider_for_dropna = [
            f for f in cols_to_consider_for_dropna if "dividend" not in f
        ]
        
        features = features.dropna(
            subset=cols_to_consider_for_dropna,
            axis=0,
        )
        print("after na", features.groupby("date").apply(len).max())

        for col in features.columns:
            if features[col].dtype == "float16":
                features[col] = features[col].astype("float32")
            gc.collect()

        gc.collect()

        save_features(features, dir_config.DAILY_PRIMARY_FEATURES_DIR)

    # gc.collect()
    # res = pd.concat(res, axis=0)

    # print("before na", res.groupby("date").apply(len).max())
    # print(res.isna().mean().sort_values())
    # cols_to_consider_for_dropna = [f for f in res.columns if "split" not in f]
    # cols_to_consider_for_dropna = [
    #     f for f in cols_to_consider_for_dropna if "dividend" not in f
    # ]

    # print(cols_to_consider_for_dropna)
    # res = res.dropna(
    #     subset=cols_to_consider_for_dropna,
    #     axis=0,
    # )
    # print("after na", res.groupby("date").apply(len).max())

    # convert float 16 to float 32 in a loop
    # for col in res.columns:
    #     if res[col].dtype == "float16":
    #         res[col] = res[col].astype("float32")
    #     gc.collect()

    # gc.collect()

    # return res


def save_features(res, DAILY_PRIMARY_FEATURES_DIR):
    print(res.head())
    # create directory if it doesn't exist
    # loop over all unique dates in chunks of 100; save each chunk to a separate file
    res["date_str"] = res["date"].dt.strftime("%Y-%m-%d")
    dates = res["date_str"].unique()

    for i in tqdm(range(0, len(dates), 200)):
        # use save_in_folders function to save each chunk to a separate folder
        _tmp = res[res["date_str"].isin(dates[i : i + 200])]
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
        print(f"primary_features_dates: {len(primary_features_dates)}")

        n_days_to_load = diff + 1000

    print(f"n_days_to_load: {n_days_to_load}")

    function_to_window: dict = {
        simple_moving_average: [5, 10, 20, 40, 50, 100, 200],
        exponential_moving_average: [5, 10, 20, 40, 50, 100, 200],
        bollinger_bands: [5, 10, 20, 50, 100, 200],
        rsi: [5, 10, 20, 40, 50, 100, 200],
        average_true_range: [5, 10, 20, 40, 50, 100, 200],
        macd: [(9, 26), (12, 26), (20, 50)],
        average_directional_index: [10, 20, 40, 50, 100],
        commodity_channel_index: [10, 20, 40, 50, 100],
        # chaikin_money_flow: [10, 20, 40, 50, 100, 200],
    }

    # load recent data
    recent_data = load_recent_data(dir_config.DAILY_DATA_DIR, n_days_to_load)
    print("recent_Data: ", recent_data.shape)
    print(recent_data.groupby("date").apply(len).max())

    generate_features(recent_data, function_to_window, USE_CUDF, dir_config)
    # features = generate_features(recent_data, function_to_window, USE_CUDF)
    # print("features_generated...")
    # print(features.shape)
    # print(features.groupby("date").apply(len).max())
    # save_features(features, dir_config.DAILY_PRIMARY_FEATURES_DIR)


if __name__ == "__main__":
    dir_config = Directories()
    generate_primary_features(dir_config)
