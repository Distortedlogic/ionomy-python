import pandas as pd
from pandas.core.frame import DataFrame

from .ionomy import Ionomy

class IonPanda(Ionomy):
    def __init__(self, **kwargs) -> None:
        Ionomy.__init__(self, **kwargs)

    def markets(self) -> DataFrame:
            return pd.DataFrame.from_records(
                super(IonPanda, self).markets()
            ).astype({
                'market': 'str',
                'title': 'str',
                'currencyBase': 'str',
                'currencyMarket': 'str',
                'orderMinSize': 'float',
                'buyFee': 'float',
                'sellFee': 'float',
                'inMaintenance': 'bool'
            })

    def currencies(self) -> DataFrame:
        return pd.DataFrame.from_records(
            super(IonPanda, self).currencies()
        ).astype({
            'currency': 'str',
            'title': 'str',
            'withdrawMinSize': 'float',
            'withdrawFee': 'float',
            'inMaintenance': 'bool',
            'canDeposit': 'bool',
            'canWithdraw': 'bool'
        })
        
    def order_book(self, market: str) -> DataFrame:
        ob = super(IonPanda, self).order_book(market)
        bids = pd.DataFrame.from_records(ob['bids'])
        asks = pd.DataFrame.from_records(ob['asks'])
        bids['type'] = 'bid'
        asks['type'] = 'ask'
        return pd.concat(
            [bids, asks]
        ).astype({
            'type': 'str',
            'size': 'float',
            'price': 'float'
        })

    def market_summaries(self) -> DataFrame:
        return pd.DataFrame.from_records(
            super(IonPanda, self).market_summaries()
        ).astype({
            'market': 'str',
            'high': 'float',
            'low': 'float',
            'volume': 'float',
            'price': 'float',
            'change': 'float',
            'baseVolume': 'float',
            'bidsOpenOrders': 'int',
            'bidsLastPrice': 'float',
            'highestBid': 'float',
            'asksOpenOrders': 'int',
            'asksLastPrice': 'float',
            'lowestAsk': 'float'
        })

    def market_history(self, market: str) -> DataFrame:
        return pd.DataFrame.from_records(
            super(IonPanda, self).market_history(market)
        ).astype({
            'type': 'str',
            'total': 'float',
            'price': 'float',
            'amount': 'float',
            'createdAt': 'datetime64'
        })

    def open_orders(self, market: str) -> DataFrame:
        return pd.DataFrame.from_records(
            super(IonPanda, self).open_orders(market)
        ).astype({
            'orderId': 'str',
            'market': 'str',
            'type': 'str',
            'amount': 'float',
            'price': 'float',
            'filled': 'float',
            'createdAt': 'datetime64'
        })

    def balances(self) -> DataFrame:
        return pd.DataFrame.from_records(
            super(IonPanda, self).balances()
        ).astype({
            'currency': 'str',
            'available': 'float',
            'reserved': 'float'
        })

    def deposit_history(self, currency: str) -> DataFrame:
        return pd.DataFrame.from_records(
            super(IonPanda, self).deposit_history(currency)
        ).astype({
            'currency': 'str',
            'deposits': 'float'
        })

    def withdrawal_history(self, currency: str) -> DataFrame:
        return pd.DataFrame.from_records(
            super(IonPanda, self).withdrawal_history(currency)['withdrawals']
        ).astype({
            'transactionId': 'str',
            'state': 'str',
            'currency': 'str',
            'amount': 'float',
            'feeAmount': 'float',
            'createdAt': 'datetime64'
        })
