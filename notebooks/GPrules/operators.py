from .rules.ret_types import bool_series, comparable_series, signals
import pandas as pd

def concat(u, v):
    signal = pd.concat([u.rename('buy'), v.rename('sell')], axis = 1)
    signal = signal.where(~(signal['buy'] & signal['sell']).astype(bool), other=0)
    return signal[(signal['buy'] == True) | (signal['sell'] == True)]

def and_(u, v):
    return u & v

def or_(u, v):
    return u | v

def not_(u):
    return ~u

def compare(u, v):
    return u.le(v)

def add_operators(pset):
    pset.addPrimitive(concat, [bool_series, bool_series], signals)
    pset.addPrimitive(and_, [bool_series, bool_series], bool_series)
    pset.addPrimitive(or_, [bool_series, bool_series], bool_series)
    pset.addPrimitive(not_, [bool_series], bool_series)
    pset.addPrimitive(compare, [comparable_series, comparable_series], bool_series)