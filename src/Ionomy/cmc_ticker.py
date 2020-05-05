from typing import Any, Dict, List, Optional, Union, overload

from requests import Session

cmc_params = {
    'start':'1',
    'limit':'100'
}

class CMCTicker(object):
    cmc_uri = 'https://pro-api.coinmarketcap.com/v1'
    cmc_latest = '/cryptocurrency/listings/latest'
    cmc_headers = {'Accepts': 'application/json'}

    def __init__(
        self,
        cmc_api_key: str,
        currency: str = "HIVE",
        base_currency: str = "BTC",
        cmc_refresh: bool = False
    ) -> None:
        self._cmc_client: Session = Session()
        self.cmc_headers['X-CMC_PRO_API_KEY'] = cmc_api_key
        self._cmc_client.headers.update(self.cmc_headers)
        self.cmc_update()
        self.cmc_refresh = cmc_refresh

    def _cmc_request(self, endpoint: str, params: dict={}) -> Any:
        resp = self._cmc_client.get(self.cmc_uri + endpoint, params=params).json()
        self.credit_count = resp['status']['credit_count']
        if not resp['status']['error_code']:
            return resp['data']
        else:
            raise Exception(resp['status']['error_message'])

    def cmc_update(self):
        cmc_params['convert'] = 'BTC'
        self.cmc_data_btc = self._cmc_request(self.cmc_latest, cmc_params)
        cmc_params['convert'] = 'USD'
        self.cmc_data_usd = self._cmc_request(self.cmc_latest, cmc_params)

    def _cmc_full(self, data: list, crypto: str = 'Hive'):
        if self.cmc_refresh:
            self.cmc_update()
        return list(filter(lambda x: x['name'] == crypto, data))[0]

    @property
    def cmc_hive_full_btc(self) -> dict:
        return self._cmc_full(self.cmc_data_btc)
    
    @property
    def cmc_hive_data_btc(self) -> dict:
        return self.cmc_hive_full_btc['quote']['BTC']

    @property
    def cmc_hive_price_btc(self) -> float:
        return self.cmc_hive_full_btc['quote']['BTC']['price']

    @property
    def cmc_hive_full_usd(self) -> dict:
        return self._cmc_full(self.cmc_data_usd)

    @property
    def cmc_hive_data_usd(self) -> dict:
        return self.cmc_hive_full_usd['quote']['USD']

    @property
    def cmc_hive_price_usd(self) -> float:
        return self.cmc_hive_full_usd['quote']['USD']['price']