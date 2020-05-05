import hashlib
import hmac
import json
from typing import Any, Dict, List, Optional, Union

import arrow
import requests
from furl import furl
from requests import Session

class Ionomy(object):
    IONOMY_URL_V1 = 'https://ionomy.com/api/v1/'

    def __init__(self, api_key: str, api_secret: str) -> None:
        self.api_calls = 0
        
        self._api_key = api_key
        self._api_secret = api_secret.encode('utf-8')

        self._ion_client: Session = requests.Session()
        self.market_names = [data["market"] for data in self._ion_request('public/markets')]
    
    def _get_signature(self, endpoint: str, params: dict, timestamp: str) -> str:
        api_furl = furl(self.IONOMY_URL_V1 + endpoint)
        api_furl.args = params
        url_ts = (api_furl.url + timestamp).encode('utf-8')
        return hmac.new(self._api_secret, url_ts, hashlib.sha512).hexdigest()

    def _ion_request(self, endpoint: str, params: dict={}):
        timestamp = str(arrow.utcnow().timestamp)
        headers = {
            'api-auth-time': timestamp,
            'api-auth-key': self._api_key,
            'api-auth-token': self._get_signature(endpoint, params, timestamp)
        }
        self.api_calls += 1
        response = self._ion_client.get(self.IONOMY_URL_V1 + endpoint, params=params, headers=headers)
        data = json.loads(response.content)
        if not data['success']:
            raise Exception(data['message'])
        return data['data']

    def markets(self) -> List[Dict[str, Any]]:
        return self._ion_request('public/markets')

    def currencies(self) -> list:
        return self._ion_request('public/currencies')
        
    def order_book(self, market: str) -> dict:
        return self._ion_request('public/orderbook', {'market': market, 'type': 'both'})

    def market_summaries(self) -> list:
        return self._ion_request('public/markets-summaries')

    def market_summary(self, market: str) -> dict:
        return self._ion_request('public/market-summary', {'market': market})

    def market_history(self, market: str) -> list:
        return self._ion_request('public/market-history', {'market': market})

    def limit_buy(
        self,
        market: str,
        amount: Union[int, float],
        price: Union[int, float]
    ) -> dict:
        params = {
            'market': market,
            'amount': f'{amount:.8f}',
            'price': f'{price:.8f}'
        }
        return self._ion_request('market/buy-limit', params)
    
    def limit_sell(
        self, market: str,
        amount: Union[int, float],
        price: Union[int, float]
    ) -> dict:
        params = {
            'market': market,
            'amount': f'{amount:.8f}',
            'price': f'{price:.8f}'
        }
        return self._ion_request('market/sell-limit', params)

    def cancel_order(self, orderId: str) -> bool:
        self._ion_request('market/cancel-order', {'orderId': orderId})
        return True

    def open_orders(self, market: str) -> list:
        return self._ion_request('market/open-orders', {'market': market})

    def balances(self) -> list:
        return self._ion_request('account/balances')

    def balance(self, currency: str) -> dict:
        return self._ion_request('account/balance', {'currency': currency})

    def deposit_address(self, currency: str) -> dict:
        return self._ion_request('account/deposit-address', {'currency': currency})

    def deposit_history(self, currency: str) -> list:
        return self._ion_request('account/deposit-history', {'currency': currency})

    def withdrawal_history(self, currency: str) -> dict:
        return self._ion_request('account/withdrawal-history', {'currency': currency})

    def get_order_status(self, orderId: str) -> dict:
        return self._ion_request('account/order', {'orderId': orderId})
