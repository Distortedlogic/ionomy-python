import arrow
import pandas as pd
from pandas.core.frame import DataFrame

from .bittrex import BitTrex
from .crypto_compare import CryptoCompare


class BitPanda(CryptoCompare, BitTrex):
    def __init__(self, api_key: str, secret_key: str) -> None:
        BitTrex.__init__(self, api_key, secret_key)

    def ohlcv(self, currency: str, base: str, time: str) -> DataFrame:
        df = pd.DataFrame.from_records(
            super(BitPanda, self).ohlcv(currency, base, time)
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
