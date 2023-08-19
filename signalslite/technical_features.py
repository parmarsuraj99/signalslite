# Would be nice to have a class for this, that automatically switches between cuda and cpu


def simple_moving_average(df, window):
    res = df["close"].rolling(window).mean()
    res.name = f"sma_{window}"
    return res


def exponential_moving_average(df, window):
    # had to adjust for cuda
    cum_sum = df["close"].rolling(window).sum()
    cum_count = df["close"].rolling(window).count()
    ema = cum_sum / cum_count
    ema.name = f"ema_{window}"
    return ema


def bollinger_bands(df, window):
    sma = simple_moving_average(df, window)
    std = df["close"].rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    upper.name = f"bbupper_{window}"
    lower.name = f"bblower_{window}"
    return upper, lower


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
