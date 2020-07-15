import pandas_ta as ta

class RSIrule:
    def __init__(self, prices, length, drift, offset, lower, upper):
        rsi = prices.rsi(length, drift, offset)
        self.signals = rsi[lower < rsi and rsi < upper].index
