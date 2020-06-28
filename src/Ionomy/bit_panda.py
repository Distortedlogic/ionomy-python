import arrow
import pandas as pd
from pandas.core.frame import DataFrame

from .bittrex import BitTrex


class BitPanda(BitTrex):
    """Pandas DataFrame Wrapper for BitTrex Base Class

    Arguments:
        api_key {str} -- BitTrex API Key
        secret_key {str} -- BitTrex API Secret
    """
    def __init__(self, api_key: str, secret_key: str) -> None:
        BitTrex.__init__(self, api_key, secret_key)

    def markets(self) -> DataFrame:
        return pd.DataFrame.from_records(
            super(BitPanda, self).markets()
        ).astype({
            'MarketCurrency': 'str',
            'BaseCurrency': 'str',
            'MarketCurrencyLong': 'str',
            'BaseCurrencyLong': 'str',
            'MinTradeSize': 'float',
            'MarketName': 'str',
            'IsActive': 'bool',
            'IsRestricted': 'bool',
            'Created': 'datetime64',
            'Notice': 'str',
            'IsSponsored': 'str',
            'LogoUrl': 'str'
        })

    def currencies(self) -> DataFrame:
        return pd.DataFrame.from_records(
            super(BitPanda, self).currencies()
        ).astype({
            'Currency': 'str',
            'CurrencyLong': 'str',
            'MinConfirmation': 'int',
            'TxFee': 'float',
            'IsActive': 'bool',
            'IsRestricted': 'bool',
            'CoinType': 'str',
            'BaseAddress': 'str',
            'Notice': 'str'
        })

    def market_summaries(self) -> DataFrame:
        return pd.DataFrame.from_records(
            super(BitPanda, self).market_summaries()
        ).astype({
            'MarketName': 'str',
            'High': 'float',
            'Low': 'float',
            'Volume': 'float',
            'Last': 'float',
            'BaseVolume': 'float',
            'TimeStamp': 'datetime64',
            'Bid': 'float',
            'Ask': 'float',
            'OpenBuyOrders': 'float',
            'OpenSellOrders': 'float',
            'PrevDay': 'float',
            'Created': 'datetime64'
        })

    def order_book(self, market: str) -> DataFrame:
        ob = super(BitPanda, self).order_book(market)
        bids = pd.DataFrame.from_records(ob['buy'])
        asks = pd.DataFrame.from_records(ob['sell'])
        bids['type'] = 'bid'
        asks['type'] = 'ask'
        return pd.concat(
            [bids, asks]
        ).astype({
            'type': 'str',
            'Quantity': 'float',
            'Rate': 'float'
        })

    def market_history(self, market: str) -> DataFrame:
        return pd.DataFrame.from_records(
            super(BitPanda, self).market_history(market)
        ).astype({
            'Id': 'int',
            'TimeStamp': 'datetime64',
            'Quantity': 'float',
            'Price': 'float',
            'Total': 'float',
            'FillType': 'str',
            'OrderType': 'str',
            'Uuid': 'str'
        })

    def balances(self) -> DataFrame:
        return pd.DataFrame.from_records(
            super(BitPanda, self).balances()
        ).astype({
            'Currency': 'str',
            'Balance': 'float',
            'Available': 'float',
            'Pending': 'float',
            'CryptoAddress': 'str'
        })

    def order_history(self) -> DataFrame:
        return pd.DataFrame.from_records(
            super(BitPanda, self).order_history()
        ).astype({
            'OrderUuid': 'str',
            'Exchange': 'str',
            'TimeStamp': 'datetime64',
            'OrderType': 'str',
            'Limit': 'float',
            'Quantity': 'float',
            'QuantityRemaining': 'float',
            'Commission': 'float',
            'Price': 'float',
            'PricePerUnit': 'float',
            'IsConditional': 'bool',
            'Condition': 'str',
            'ConditionTarget': 'float',
            'ImmediateOrCancel': 'bool',
            'Closed': 'datetime64'
        })

    def deposit_history(self, currency: str) -> DataFrame:
        data = super(BitPanda, self).deposit_history(currency)
        if not data:
            return pd.DataFrame(data={
                'Id': [],
                'Amount': [],
                'Currency': [],
                'Confirmations': [],
                'LastUpdated': [],
                'TxId': [],
                'CryptoAddress': []
            })
        return pd.DataFrame.from_records(
            data
        ).astype({
            'Id': 'int',
            'Amount': 'float',
            'Currency': 'str',
            'Confirmations': 'int',
            'LastUpdated': 'datetime64',
            'TxId': 'str',
            'CryptoAddress': 'str'
        })

    def withdrawal_history(self, currency: str) -> DataFrame:
        data = super(BitPanda, self).withdrawal_history(currency)
        if not data:
            return pd.DataFrame(data={
                "PaymentUuid": [],
                "Currency": [],
                "Amount": [],
                "Address": [],
                "Opened": [],
                "Authorized": [],
                "PendingPayment": [],
                "TxCost": [],
                "TxId": [],
                "Canceled": [],
                "InvalidAddress": []
            })
        return pd.DataFrame.from_records(
            data
        ).astype({
            "PaymentUuid": "str",
            "Currency": "str",
            "Amount": 'float',
            "Address": "str",
            "Opened": "datetime64",
            "Authorized": "bool",
            "PendingPayment": "bool",
            "TxCost": 'float',
            "TxId": "str",
            "Canceled": "bool",
            "InvalidAddress": "bool"
        })

    def ohlcv(self, currency: str, base: str, time: str, limit: int) -> DataFrame:
        df = pd.DataFrame.from_records(
            super(BitPanda, self).ohlcv(currency, base, time, limit)
        ).drop(
            columns=['conversionType', 'conversionSymbol']
        ).rename(
            columns={
                'volumefrom': f'volume{currency.lower()}',
                'volumeto': 'volume'
            }
        )
        df['date'] = df['time'].apply(lambda ts: arrow.get(ts).format('YYYY-MM-DD'))
        return df
