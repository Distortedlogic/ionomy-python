import random

from pandas.core.frame import DataFrame
from .ret_types import bool_series, comparable_series

class diff_lower:
    pass
class diff_upper:
    pass

price_terminals = [diff_lower, diff_upper]

def price(u):
    return u['close']

def percent_diff_lt(u, diff_lower):
    return u['close'].pct_change() < diff_lower

def percent_diff_gt(u, diff_upper):
    return u['close'].pct_change() > diff_upper

def add_price_rule(pset):
    pset.addPrimitive(price, [DataFrame], comparable_series)

    pset.addEphemeralConstant("diff_lower", lambda: random.uniform(-1, 0), diff_lower)
    pset.addEphemeralConstant("diff_upper", lambda: random.uniform(0, 1), diff_upper)

    pset.addPrimitive(percent_diff_lt, [DataFrame, diff_lower], bool_series)
    pset.addPrimitive(percent_diff_gt, [DataFrame, diff_upper], bool_series)
