from typing import Optional

import pandas as pd
import pandas_ta as ta
from pandas.core.frame import DataFrame, Series

from .ion_panda import IonPanda

class IonTA(IonPanda):

    def __init__(self, **kwargs) -> None:
        IonPanda.__init__(self, **kwargs)

    def update(self, crypto: str = 'HIVE', base: str = 'BTC'):
        self.df = self.ohlcv(crypto, base)

    @staticmethod
    def _size_mask(
        df: DataFrame,
        size_min: Optional[float] = None,
        size_max: Optional[float] = None
    ) -> Series:
        if size_min and size_max:
            return (df['size']>=size_min) & (df['size']<=size_max)
        elif size_min:
            return (df['size']>=size_min)
        elif size_max:
            return (df['size']<=size_max)
        else:
            return [True] * len(df)

    def _filter_orderbook(
        self,
        market: str,
        order_type: str,
        size_min: Optional[float] = None,
        size_max: Optional[float] = None
    ):
        order_book_pd = self.order_book(market)
        order_book_pd = order_book_pd[order_book_pd['type']==order_type]
        mask = self._size_mask(order_book_pd, size_min, size_max)
        
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
