# Would be nice to have a class for this, that automatically switches between cuda and cpu


def simple_moving_average(df, window):
    res = df["close"].rolling(window).mean()
    res.name = f"sma_{window}"
    return res


def exponential_moving_average(df, window):
    # had to adjust for cuda; cudf doesn't have .ewm() yet
    cum_sum = df["close"].rolling(window).sum()
    cum_count = df["close"].rolling(window).count()
    ema = cum_sum / cum_count
    ema.name = f"ema_{window}"
    return ema


def bollinger_bands(df, window):
    sma = df["close"].rolling(window).mean().ffill().bfill()
    std = df["close"].rolling(window).std().ffill().bfill()
    upper = sma + 2 * std
    lower = sma - 2 * std

    # ratio
    upper = upper / df["close"] - 1
    lower = lower / df["close"] - 1

    upper.name = f"bbupper_close_{window}"
    lower.name = f"bblower_close_{window}"
    bb_width = upper - lower
    bb_width.name = f"bbwidth_close_{window}"
    return upper, lower, bb_width


def rsi(df, window):
    delta = df["close"].diff()
    up_days = delta.copy()
    up_days[delta <= 0] = 0.0
    down_days = abs(delta.copy())
    down_days[delta > 0] = 0.0
    RS_up = up_days.rolling(window).mean()
    RS_down = down_days.rolling(window).mean()
    rsi = 100 - 100 / (1 + RS_up / RS_down)
    rsi.name = f"rsi_{window}"
    return rsi


def macd(df, window_fast, window_slow):
    ema_fast = exponential_moving_average(df, window_fast)
    ema_slow = exponential_moving_average(df, window_slow)
    macd = ema_fast - ema_slow
    macd.name = f"macd_{window_fast}_{window_slow}"
    return macd


# ATR
def average_true_range(df, window):
    tr = df[["high", "low", "close"]].max(axis=1) - df[["high", "low", "close"]].min(
        axis=1
    )
    atr = tr.rolling(window).mean()
    atr.name = f"atr_{window}"
    return atr

# cmf
def chaikin_money_flow(df, window):
    # Calculate the Money Flow Multiplier
    mf_multiplier = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"])

    # Calculate the Money Flow Volume
    mf_volume = mf_multiplier * df["volume"]
    
    # Calculate the CMF Line
    cmf_line = mf_volume.rolling(window).sum() / df["volume"].rolling(window).sum()
    
    cmf_line.name = f"cmf_{window}"
    
    return cmf_line

def average_directional_index(df, window):
    tr = df["high"] - df["low"]
    dm_plus = (df["high"].diff() > df["low"].diff()) * df["high"].diff()
    dm_minus = (df["low"].diff() > df["high"].diff()) * df["low"].diff()
    tr_smoothed = tr.rolling(window).sum()
    dm_plus_smoothed = dm_plus.rolling(window).sum()
    dm_minus_smoothed = dm_minus.rolling(window).sum()
    di_plus = (dm_plus_smoothed / tr_smoothed) * 100
    di_minus = (dm_minus_smoothed / tr_smoothed) * 100
    dx = ((di_plus - di_minus).abs() / (di_plus + di_minus)) * 100
    adx = dx.rolling(window).mean()
    adx.name = f"adx_{window}"
    return adx

def commodity_channel_index(df, window):
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    mean_deviation = typical_price.rolling(window).std()
    
    sma_typical_price = typical_price.rolling(window).mean()
    cci = (typical_price - sma_typical_price) / (0.015 * mean_deviation)
    cci.name = f"cci_{window}"
    
    return cci
