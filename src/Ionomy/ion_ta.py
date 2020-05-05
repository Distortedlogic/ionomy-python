from typing import Optional
from Ionomy.ion_panda import IonPanda
import pandas as pd
import pandas_ta as ta

class IonTA(IonPanda):
    default_market = 'btc-hive'
    default_currency = 'hive'
    default_base = 'btc'
    def __init__(self, api_key: str, api_secret: str) -> None:
        IonPanda.__init__(self, api_key, api_secret)

    def max_bid(
        self,
        market: str,
        size_min: Optional[float] = None,
        size_max: Optional[float] = None
    ) -> float:
        order_book_pd = self.order_book(market)
        order_book_pd = order_book_pd[order_book_pd['type']=='bid']
        mask = [True] * len(order_book_pd)
        if size_min and size_max:
            mask = (order_book_pd['size']>=size_min) & (order_book_pd['size']<=size_max)
        elif size_min:
            mask = (order_book_pd['size']>=size_min)
        elif size_max:
            mask = (order_book_pd['size']<=size_max)
        
        return order_book_pd[mask]['price'].max()
    
    def min_ask(
        self,
        market: str,
        size_min: Optional[float] = None,
        size_max: Optional[float] = None
    ) -> float:
        order_book_pd = self.order_book(market)
        order_book_pd = order_book_pd[order_book_pd['type']=='ask']
        mask = [True] * len(order_book_pd)
        if size_min and size_max:
            mask = (order_book_pd['size']>=size_min) & (order_book_pd['size']<=size_max)
        elif size_min:
            mask = (order_book_pd['size']>=size_min)
        elif size_max:
            mask = (order_book_pd['size']<=size_max)
        
        return order_book_pd[mask]['price'].min()

    def spread(self, market: str = 'HIVE'):
        return self.min_ask(market) - self.max_bid(market)

    