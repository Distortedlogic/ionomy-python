from typing import Callable, Dict, List, Union
import numpy as np
import pandas as pd
import pandas_ta as ta
from pandas.core.frame import DataFrame
from pandas.core.series import Series

EMA_WINDOW = 9

def diff_percent(base: Series, other: Series) -> Series:
    return base.subtract(other, fill_value=0).divide(base, fill_value=0)

class Environment:
    def __init__(
        self,
        ohlcv_df: DataFrame,
        state_creators: List[Dict[str, Union[str, bool, int, float]]],
        length: int = 250
    ) -> None:
        self.ohlcv_df = ohlcv_df[-length:]
        self.close = ohlcv_df["close"]
        self.length = len(self.close) - 1
        self.state_creators = state_creators
        self.dim = len(self.state_creators)+1

    def _state(
        self,
        time_index,
        window_size,
        name,
        **kwargs
    ) -> np.ndarray:
        super_state = diff_percent(self.close, getattr(self.ohlcv_df.ta, name)(**kwargs))
        return super_state.loc[time_index+1-window_size:time_index+1].to_numpy()

    def get_state(self, time_index: int, window_size: int) -> np.ndarray:
        final = self.close.pct_change().loc[time_index+1-window_size:time_index+1].to_numpy(copy=True)
        states = np.array([
            self._state(
                time_index,
                window_size,
                **state_creator
            ) for state_creator in self.state_creators
        ])
        return np.vstack(final, states).flatten()
