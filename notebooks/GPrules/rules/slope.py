import random

import pandas_ta as ta

from pandas.core.frame import DataFrame
from .ret_types import bool_series, comparable_series, filterable_series

class slope_length:
    pass
class slope_offset:
    pass
class slope_upper:
    pass
class slope_lower:
    pass


slope_terminals = [slope_length, slope_offset, slope_upper, slope_lower]

def slope(u, slope_length, slope_offset):
    return u.ta.slope(slope_length, slope_offset)

def slope_gt(u, slope_length, slope_offset, slope_upper):
    return u.ta.slope(slope_length, slope_offset) > slope_upper

def slope_lt(u, slope_length, slope_offset, slope_lower):
    return u.ta.slope(slope_length, slope_offset) < slope_lower

def add_slope_rule(pset):
    pset.addEphemeralConstant("slope_length", lambda: random.randint(0, 50), slope_length)
    pset.addEphemeralConstant("slope_offset", lambda: random.randint(0, 10), slope_offset)

    pset.addEphemeralConstant("slope_lower", lambda: random.uniform(-1, 0), slope_lower)
    pset.addEphemeralConstant("slope_upper", lambda: random.uniform(0, 1), slope_upper)

    pset.addPrimitive(slope_lt, [DataFrame, slope_length, slope_offset, slope_lower], filterable_series)
    pset.addPrimitive(slope_gt, [DataFrame, slope_length, slope_offset, slope_upper], filterable_series)
