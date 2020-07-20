import pandas as pd

def series_concat(u, v):
    return pd.concat([u.rename('buy'), v.rename('sell')], axis = 1)

def series_and(u, v):
    return u & v

def series_or(u, v):
    return u | v

def series_not(u):
    return ~u

def series_lt(u, v):
    return u.le(v)

def series_lt_float(u, v):
    return u < v

def series_gt_float(u, v):
    return u > v

##################################################################################################

def price(u):
    return u['close']

def percent_diff(u, v):
    if u.eq(v['close']):
        return u.pct_change()
    return v['close'].substract(u).divide(v['close'])