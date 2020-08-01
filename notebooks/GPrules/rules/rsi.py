from pandas.core.frame import DataFrame
import random
import pandas as pd
import pandas_ta as ta

from .ret_types import bool_series, filterable_series

class rsi_length:
    pass
class rsi_drift:
    pass
class rsi_offset:
    pass
class rsi_lower:
    pass
class rsi_upper:
    pass

rsi_terminals = [rsi_length, rsi_drift, rsi_offset, rsi_lower, rsi_upper]

def rsi_lt(df, length, drift, offset, threshold):
    return df.ta.rsi(length = length, drift = drift, offset = offset) < threshold

def rsi_gt(df, length, drift, offset, threshold):
    return df.ta.rsi(length = length, drift = drift, offset = offset) > threshold

def add_rsi_rule(pset):
    pset.addEphemeralConstant("rsi_length", lambda: random.randint(1, 250), rsi_length)
    pset.addEphemeralConstant("rsi_drift", lambda: random.randint(0, 10), rsi_drift)
    pset.addEphemeralConstant("rsi_offset", lambda: random.randint(0, 10), rsi_offset)
    pset.addEphemeralConstant("rsi_lower", lambda: random.randint(10, 40), rsi_lower)
    pset.addEphemeralConstant("rsi_upper", lambda: random.randint(60, 90), rsi_upper)

    pset.addPrimitive(rsi_lt, [DataFrame, rsi_length, rsi_drift, rsi_offset, rsi_lower], filterable_series)
    pset.addPrimitive(rsi_gt, [DataFrame, rsi_length, rsi_drift, rsi_offset, rsi_upper], filterable_series)