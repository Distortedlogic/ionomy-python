from pandas.core.frame import DataFrame
import random
import pandas as pd
import pandas_ta as ta

from .ret_types import bool_series, comparable_series, filterable_series

class ema_length:
    pass
class ema_offset:
    pass
class ema_lower:
    pass
class ema_upper:
    pass

ema_terminals = [ema_length, ema_offset, ema_lower, ema_upper]

def ema(df, length, offset):
    return df.ta.ema(length = length, offset = offset)
def ema_lt(df, length, offset, ema_lower):
    ema = df.ta.ema(length = length, offset = offset)
    prices = df['close']
    diff = prices.sub(ema).divide(prices)
    return (diff < ema_lower).astype(int)
def ema_gt(df, length, offset, ema_upper):
    ema = df.ta.ema(length = length, offset = offset)
    prices = df['close']
    diff = prices.sub(ema).divide(prices)
    return diff > ema_upper

def add_ema_rule(pset):
    pset.addEphemeralConstant("ema_length", lambda: random.randint(1, 250), ema_length)
    pset.addEphemeralConstant("ema_offset", lambda: random.randint(0, 10), ema_offset)
    pset.addPrimitive(ema, [DataFrame, ema_length, ema_offset], comparable_series)

    pset.addEphemeralConstant("ema_lower", lambda: random.uniform(-0.01, 0), ema_lower)
    pset.addEphemeralConstant("ema_upper", lambda: random.uniform(0, 0.01), ema_upper)
    pset.addPrimitive(ema_lt, [DataFrame, ema_length, ema_offset, ema_lower], filterable_series)
    pset.addPrimitive(ema_gt, [DataFrame, ema_length, ema_offset, ema_upper], filterable_series)