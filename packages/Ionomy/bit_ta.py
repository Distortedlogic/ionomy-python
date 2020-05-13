import pandas as pd
import pandas_ta as ta

from typing import Optional
from pandas.core.frame import DataFrame, Series

from .bit_panda import BitPanda
from .utils.dataframes import _size_mask

class BitTA(BitPanda):
    def __init__(self, **kwargs) -> None:
        BitPanda.__init__(self, **kwargs)

    def update(self, crypto: str, base: str, time: str):
        self.df = self.ohlcv(crypto, base, time)

    def _filter_orderbook(
        self,
        market: str,
        order_type: str,
        size_min: Optional[float] = None,
        size_max: Optional[float] = None
    ):
        order_book_pd = self.order_book(market)
        order_book_pd = order_book_pd[order_book_pd['type']==order_type]
        mask = _size_mask(order_book_pd, size_min, size_max)
        
        return order_book_pd[mask]

    def max_bid(
        self,
        market: str,
        size_min: Optional[float] = None,
        size_max: Optional[float] = None
    ) -> float:
        return self._filter_orderbook(market, 'bid', size_min, size_max)['price'].max()
    
    def min_ask(
        self,
        market: str,
        size_min: Optional[float] = None,
        size_max: Optional[float] = None
    ) -> float:
        return self._filter_orderbook(market, 'ask', size_min, size_max)['price'].min()

    def spread(self, market: str = 'HIVE'):
        return self.min_ask(market) - self.max_bid(market)

    def rsi(self, length=None, drift=None, offset=None):
        return self.df.ta.rsi(length=length, drift=drift, offset=offset)

    def macd(self, fast=None, slow=None, signal=None, offset=None):
        return self.df.ta.macd(fast=fast, slow=slow, signal=signal, offset=offset)

    def momentum(self, length=None, offset=None):
        return self.df.ta.mom(length=length, offset=offset)

    def roc(self, length=None, offset=None):
        return self.df.ta.roc(length=length, offset=offset)

    def sma(self, length=None, offset=None):
        return self.df.ta.ema(length=length, offset=offset)

    def ema(self, length=None, offset=None):
        return self.df.ta.ema(length=length, offset=offset)

    def vwma(self, length=None, offset=None):
        return self.df.ta.vwma(length=length, offset=offset)

    def atr(self, length=None, mamode=None, offset=None):
        return self.df.ta.atr(length=None, mamode=None, offset=None)

    def log_return(self, length=None, cumulative=False, percent=False, offset=None):
        return self.df.ta.log_return(length=length, cumulative=cumulative, percent=percent, offset=offset)

    def percent_return(self, length=None, cumulative=False, percent=False, offset=None):
        return self.df.ta.percent_return(length=length, cumulative=cumulative, percent=percent, offset=offset)

    