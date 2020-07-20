from pandas.core.frame import DataFrame
import random
import pandas as pd
import pandas_ta as ta

from .ret_types import percent_series

class rsi_length:
    pass
class rsi_drift:
    pass
class rsi_offset:
    pass

def rsi(df, length, drift, offset):
    return df.ta.rsi(length = length, drift = drift, offset = offset)/100

def add_rsi(pset):
    pset.addEphemeralConstant("rsi_length", lambda: random.randint(1, 250), rsi_length)
    pset.addEphemeralConstant("rsi_drift", lambda: random.randint(0, 25), rsi_drift)
    pset.addEphemeralConstant("rsi_offset", lambda: random.randint(0, 25), rsi_offset)
    
    pset.addPrimitive(rsi, [DataFrame, rsi_length, rsi_drift, rsi_offset], percent_series)