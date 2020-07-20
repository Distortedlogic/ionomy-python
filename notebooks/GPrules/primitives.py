from pandas.core.frame import DataFrame
import pandas
import pandas as pd
import pandas_ta as ta

import random

from deap import gp, creator, base, tools

from .operators import percent_diff, price, series_and, series_concat, series_gt_float, series_lt, series_lt_float, series_not, series_or
from .rules.ret_types import bool_series, comparable_series, percent_series, signals

from .rules.rsi import add_rsi, rsi_drift, rsi_length, rsi_offset
from .rules.ema import add_ema, ema_length, ema_offset

def build_pset():
    pset = gp.PrimitiveSetTyped("MAIN", [DataFrame], signals)
    pset.renameArguments(ARG0="df")
    pset.addEphemeralConstant("rand_float", lambda: random.random(), float)

    pset.addPrimitive(series_concat, [bool_series, bool_series], signals)
    pset.addPrimitive(series_and, [bool_series, bool_series], bool_series)
    pset.addPrimitive(series_or, [bool_series, bool_series], bool_series)
    pset.addPrimitive(series_not, [bool_series], bool_series)
    pset.addPrimitive(series_lt, [comparable_series, comparable_series], bool_series)
    pset.addPrimitive(series_lt_float, [percent_series, float], bool_series)
    pset.addPrimitive(series_gt_float, [percent_series, float], bool_series)
    pset.addPrimitive(price, [DataFrame], comparable_series)
    pset.addPrimitive(percent_diff, [comparable_series, DataFrame], percent_series)

    add_rsi(pset)
    add_ema(pset)

    terminal_types = [
        pandas.core.frame.DataFrame,
        float,
        ema_length,
        ema_offset,
        rsi_length,
        rsi_drift,
        rsi_offset
    ]

    return pset, terminal_types

