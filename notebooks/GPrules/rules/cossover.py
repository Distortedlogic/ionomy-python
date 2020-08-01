from pandas.core.frame import DataFrame
import random
import pandas as pd
import pandas_ta as ta

from .ret_types import bool_series, comparable_series, not_able_series

class macd_fast:
    pass
class macd_slow:
    pass
class macd_signal:
    pass
class macd_offset:
    pass

macd_terminals = [macd_fast, macd_slow, macd_signal, macd_offset]

def crossover(u, v):
    neg = u.le(v)
    pos = u.gt(v)
    return pos.shift(1) & neg

def macd_crossover(df, macd_fast, macd_slow, macd_signal, macd_offset):
    if macd_fast > macd_slow:
        macd_fast, macd_slow = macd_slow, macd_fast
    elif macd_fast == macd_slow:
        macd_slow += 1
    macd_full = df.ta.macd(fast=macd_fast, slow=macd_slow, signal=macd_signal, offset=macd_offset)
    macdh = macd_full.iloc[:, 1]

    return crossover(macdh, 0)

def add_crossover_rule(pset):
    pset.addPrimitive(crossover, [comparable_series, comparable_series], not_able_series)

    pset.addEphemeralConstant("macd_fast", lambda: random.randint(1, 250), macd_fast)
    pset.addEphemeralConstant("macd_slow", lambda: random.randint(1, 250), macd_slow)
    pset.addEphemeralConstant("macd_signal", lambda: random.randint(1, 250), macd_signal)
    pset.addEphemeralConstant("macd_offset", lambda: random.randint(0, 10), macd_offset)
    pset.addPrimitive(macd_crossover, [DataFrame, macd_fast, macd_slow, macd_signal, macd_offset], not_able_series)