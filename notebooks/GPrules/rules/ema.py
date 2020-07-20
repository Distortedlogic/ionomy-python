from pandas.core.frame import DataFrame
import random
import pandas as pd
import pandas_ta as ta

from .ret_types import comparable_series

class ema_length:
    pass
class ema_offset:
    pass

def ema(df, length, offset):
    return df.ta.ema(length = length, offset = offset)

def add_ema(pset):
    pset.addEphemeralConstant("ema_length", lambda: random.randint(1, 250), ema_length)
    pset.addEphemeralConstant("ema_offset", lambda: random.randint(0, 25), ema_offset)
    pset.addPrimitive(ema, [DataFrame, ema_length, ema_offset], comparable_series)