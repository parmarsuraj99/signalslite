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

# parallelize the process on all columns using joblib
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import List, Tuple, Union, Optional, Dict, Any

from signalslite.data_utils import (
    load_recent_data_from_file,
    save_daily_data,
    save_in_folders,
    get_latest_date,
    read_available_dates,
)
from signalslite.constants import Directories


def apply_cut(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    # NOTE: Parallel is a library for parallelizing operations
    # on a single machine. It is not a distributed framework.
    # The `n_jobs` argument specifies how many CPU cores to use.
    # For example, `n_jobs=10` will use 10 CPU cores.
    _res = Parallel(
        n_jobs=10,
    )(delayed(pd.qcut)(df[col], q=5, labels=False, duplicates="drop") for col in cols)
    _res = pd.concat(_res, axis=1).astype("int8")
    return _res


def apply_cut_cpu(
    recent_data: pd.DataFrame, feature_columns: List[str]
) -> pd.DataFrame:
    recent_data = recent_data.dropna(subset=feature_columns, axis=0)
    recent_data_gpu = cudf.DataFrame.from_pandas(recent_data)

    ranks = (
        recent_data_gpu[["date"] + feature_columns]
        .groupby("date")
        .rank(pct=True, method="first", ascending=True, na_option="keep")
    )
    ranks["close"] = recent_data["close"]
    ranks["date"] = recent_data["date"]
    ranks["bloomberg_ticker"] = recent_data["bloomberg_ticker"]

    ranks_pd = ranks.to_pandas()

    del recent_data_gpu, ranks

    # NOTE: This code will be run on the CPU.
    # We will show you how to run this code on the GPU in the next section.
    res = ranks_pd.groupby("date").apply(lambda df: apply_cut(df, feature_columns))
    res["date"] = ranks_pd["date"]
    res["bloomberg_ticker"] = ranks_pd["bloomberg_ticker"]
    res["close"] = ranks_pd["close"]

    return res


# this code is used to apply a cut on the data, in order to remove some of the data that has been corrupted
# the cut is based on the standard deviation of the data, and is applied on a per-feature basis
# it is applied to the data in chunks of 200 days, in order to avoid memory issues
def scale_data(dir_config, FROM_SCRATCH=False):
    dates = read_available_dates(dir_config.DAILY_SECONDARY_FEATURES_DIR)

    start_index = len(dates) - 1000 if not FROM_SCRATCH else 0

    # iterate over all dates in chunks of 200
    for i in tqdm(range(start_index, len(dates), 200)):
        _tmp = load_recent_data_from_file(
            dir_config.DAILY_SECONDARY_FEATURES_DIR,
            n_days=200,
            ascending=True,
            offset=i,
            dtype="float32",
        )
        _tmp = _tmp.reset_index(drop=True)
        _tmp = _tmp.sort_values(["date", "bloomberg_ticker"])
        _tmp = _tmp.groupby("date").filter(lambda x: len(x) > 10)

        # get all feature columns
        feature_columns = [f for f in _tmp.columns if f.startswith("feature_")]

        # apply cut
        res = apply_cut_cpu(_tmp, feature_columns)

        # save
        save_in_folders(res, dir_config.DAILY_SCALED_FEATURES_DIR)

        del _tmp, res
        gc.collect()


if __name__ == "__main__":
    dir_config = Directories()
    # dir_config.set_data_dir("data")

    FROM_SCRATCH = False

    scale_data(dir_config, FROM_SCRATCH=FROM_SCRATCH)
