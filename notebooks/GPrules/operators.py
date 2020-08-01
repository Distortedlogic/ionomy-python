import pandas as pd
import pandas
from pandas.core.frame import DataFrame
import random
from .rules.ret_types import bool_series, comparable_series, filterable_series, not_able_series, signals

class min_hold:
    pass

operator_terminals = [pandas.core.frame.DataFrame, min_hold]

def concat(u, v, df, min_hold):
    signal = pd.concat([u.rename('buy'), v.rename('sell')], axis = 1)
    signal = signal.where(~(signal['buy'] & signal['sell']).astype(bool), other=0)
    signal = pd.concat([signal, df['close']], axis=1)
    signal = signal[250:].reset_index(drop=True)
    signal = signal[(signal['buy'] == True) | (signal['sell'] == True)]
    return signal, min_hold

def and_(u, v):
    return u & v

def or_(u, v):
    return u | v

def not_(u):
    return ~u

def compare(u, v):
    return u.lt(v)

def add_operators(pset):
    pset.addEphemeralConstant("min_hold", lambda: random.randint(5, 48), min_hold)
    pset.addPrimitive(concat, [bool_series, bool_series, DataFrame, min_hold], signals)
    pset.addPrimitive(and_, [bool_series, bool_series], not_able_series)
    pset.addPrimitive(or_, [bool_series, bool_series], not_able_series)
    pset.addPrimitive(not_, [not_able_series], bool_series)
    pset.addPrimitive(compare, [comparable_series, comparable_series], filterable_series)
