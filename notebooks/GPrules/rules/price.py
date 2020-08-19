import random

import pandas_ta as ta

from pandas.core.frame import DataFrame
from .ret_types import bool_series, comparable_series, filterable_series

class vwap_offset:
    pass
class diff_lower:
    pass
class diff_upper:
    pass

price_terminals = [diff_lower, diff_upper, vwap_offset]

def price(u):
    return u['close']

def vwap(u, vwap_offset):
    return u.ta.vwap(offset=vwap_offset)

def percent_diff_vwap_lt(u, vwap_offset, diff_lower):
    return u['close'].sub(u.ta.vwap(offset=vwap_offset)).divide(u['close']) < diff_lower

def percent_diff_vwap_gt(u, vwap_offset, diff_upper):
    return u['close'].sub(u.ta.vwap(offset=vwap_offset)).divide(u['close']) > diff_upper

def percent_diff_lt(u, diff_lower):
    return u['close'].pct_change() < diff_lower

def percent_diff_gt(u, diff_upper):
    return u['close'].pct_change() > diff_upper

def add_price_rule(pset):
    pset.addEphemeralConstant("vwap_offset", lambda: random.randint(0, 10), vwap_offset)

    pset.addEphemeralConstant("diff_lower", lambda: random.uniform(-1, 0), diff_lower)
    pset.addEphemeralConstant("diff_upper", lambda: random.uniform(0, 1), diff_upper)

    pset.addPrimitive(price, [DataFrame], comparable_series)
    pset.addPrimitive(vwap, [DataFrame, vwap_offset], comparable_series)

    pset.addPrimitive(percent_diff_lt, [DataFrame, diff_lower], filterable_series)
    pset.addPrimitive(percent_diff_gt, [DataFrame, diff_upper], filterable_series)

    pset.addPrimitive(percent_diff_vwap_lt, [DataFrame, vwap_offset, diff_lower], filterable_series)
    pset.addPrimitive(percent_diff_vwap_gt, [DataFrame, vwap_offset, diff_upper], filterable_series)
