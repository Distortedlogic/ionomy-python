import numpy as np
import pandas as pd
import pandas_ta as ta
from pandas.core.frame import DataFrame
from pandas.core.series import Series

def diff_percent(base: Series, other: Series) -> Series:
    return other.subtract(base, fill_value=0).divide(base, fill_value=0)

class Environment:
    def __init__(
        self,
        df: DataFrame,
        window_size: int,
        initial_capital: int,
        max_buy: float,
        max_sell: float,
        length: float = 750
    ) -> None:
        ema_10 = df.ta.ema(length=10).sub(df['close']).divide(df['close']).rename('ema_10')
        ema_50 = df.ta.ema(length=50).sub(df['close']).divide(df['close']).rename('ema_50')
        ema_100 = df.ta.ema(length=100).sub(df['close']).divide(df['close']).rename('ema_100')
        ema_200 = df.ta.ema(length=200).sub(df['close']).divide(df['close']).rename('ema_200')
        true_range = df.ta.true_range().divide(df['close']).rename('true_range')
        pct_change = df['close'].pct_change().rename('pct_change')
        self.df = pd.concat([pct_change, ema_10, ema_50, ema_100, ema_200, true_range], axis=1)
        self.df = self.df.iloc[length:].reset_index(drop=True)
        self.num_features = len(self.df.columns)
        self.close = df["close"].iloc[length:].reset_index(drop=True)
        self.length = len(self.close) - 1
        self.window_size = window_size
        self.initial_capital = initial_capital
        self.max_buy = max_buy
        self.max_sell = max_sell
        
    def get_state(self, time_index: int) -> np.ndarray:
        return self.df.loc[time_index+2-self.window_size:time_index+1].to_numpy(copy=True).flatten()
