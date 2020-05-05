from typing import Any, Dict, List, Optional, Union, overload

from requests import Session

class CCTicker(object):
    crypto_compare_uri = 'https://min-api.cryptocompare.com/data'
    hourly_ohlcv = '/v2/histohour'
    daily_ohlcv = '/v2/histoday'
    cc_price_endpoint = '/price'

    def __init__(
        self,
        currency: str = "HIVE",
        base_currency: str = "BTC",
        cmc_api_key: str = '',
        cmc_refresh: bool = False
    ) -> None:
        self._cc_client: Session = Session()

    def _cc_request(self, endpoint: str, params: dict = {}) -> Any:
        return self._cc_client.get(self.crypto_compare_uri + endpoint, params=params).json()
    
    def cc_spot_price(
        self,
        crypto: str = "HIVE",
        base: str = "BTC",
    ) -> float:
        params = {"fsym": crypto, "tsyms": base}
        return self._cc_request(self.cc_price_endpoint, params)[base]

    def cc_ohlcv(self, crypto: str = 'HIVE', base: str = 'BTC'):
        params = {"fsym": crypto, "tsym": base}
        resp = self._cc_request(self.daily_ohlcv, params)
        if resp['Response'] == 'Success':
            self.start_time = resp['Data']['TimeFrom']
            self.end_time = resp['Data']['TimeTo']
            return resp['Data']['Data']
        else:
            raise Exception(resp['Message'])