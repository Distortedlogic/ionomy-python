import numpy as np
from pandas.core.frame import DataFrame
from pandas.core.series import Series

def diff_percent(base: Series, other: Series) -> Series:
    return other.subtract(base, fill_value=0).divide(base, fill_value=0)

class Environment:
    def __init__(
        self,
        ohlcv_df: DataFrame,
        window_size: int,
        initial_capital: int,
        max_buy: float,
        max_sell: float,
        length: float = 250
    ) -> None:
        self.ohlcv_df = ohlcv_df
        self.close = self.ohlcv_df["close"]
        self.length = len(self.close) - 1
        self.window_size = window_size
        self.initial_capital = initial_capital
        self.max_buy = max_buy
        self.max_sell = max_sell
        self.config = {
            "window_size": window_size,
            "initial_capital": initial_capital,
            "max_buy": max_buy,
            "max_sell": max_sell
        }
        
    def get_state(self, time_index: int) -> np.ndarray:
        return self.close.pct_change().loc[time_index+2-self.window_size:time_index+1].to_list()
