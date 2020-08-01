from pandas.core.frame import DataFrame
import random
import pandas as pd
import pandas_ta as ta

from .ret_types import bool_series, filterable_series, not_able_series

class filter_int:
    pass

filter_terminals = [filter_int]

def signal_filter(u, filter_int_1, filter_int_2):
    if filter_int_1 < filter_int_2:
        filter_window = filter_int_2
        filter_trigger = filter_int_1
    else:
        filter_window = filter_int_1
        filter_trigger = filter_int_2
    return u.rolling(filter_window).sum() >= filter_trigger

def add_filter_rule(pset):
    pset.addEphemeralConstant("filter_int", lambda: random.randint(2, 10), filter_int)
    pset.addPrimitive(signal_filter, [filterable_series, filter_int, filter_int], not_able_series)