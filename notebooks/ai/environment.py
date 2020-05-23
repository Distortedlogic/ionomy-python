import numpy as np
import pandas as pd
import pandas_ta as ta
from pandas.core.frame import DataFrame
from pandas.core.series import Series

EMA_WINDOW = 9

def price_change_state(data: np.array, time_index: int, window_size: int) -> np.array:
    d = time_index - window_size + 1
    block = data[d : time_index + 1] if d >= 0 else -d * [data[0]] + data[0 : time_index + 1]
    return np.array([[block[i + 1] - block[i] for i in range(window_size - 1)]])

def diff_percent(base: Series, other: Series) -> Series:
    return base.subtract(other, fill_value=0).divide(base, fill_value=0)

class Environment:
    def __init__(self, ohlcv_df: DataFrame, length: int = 250):
        self.ohlcv_df = ohlcv_df
        close_series = ohlcv_df.loc[-length:,"close"]
        ema_series = ohlcv_df.ta.ema(length=EMA_WINDOW).loc[-length:]
        self.ema_diff_percent = diff_percent(close_series, ema_series)
        self.close = close_series.to_list()
        self.ema = ema_series.to_list()
        self.length = len(self.close) - 1

    @property
    def shape(self):
        return
