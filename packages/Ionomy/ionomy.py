import hashlib
import hmac
import json
import arrow
import requests

from furl import furl
from requests import Session
from typing import Any, Dict, List, Optional, Union, Callable, Any

from .ticker import Ticker
from .utils.decorators import currency_to_crypto

class Ionomy(Ticker):
    _ion_client: Session = requests.Session()

    def __init__(self, ion_api_key: str, ion_api_secret: str, **kwargs) -> None:
        Ticker.__init__(self, **kwargs)
        self.ion_api_key = ion_api_key
        self.ion_api_secret = ion_api_secret
        self.market_names = [data["market"] for data in self._ion_request('public/markets')]
    
    def _get_signature(self, endpoint: str, params: dict, timestamp: str) -> str:
        api_furl = furl('https://ionomy.com/api/v1/' + endpoint)
        api_furl.args = params
        url_ts = (api_furl.url + timestamp).encode('utf-8')
        return hmac.new(
            self.ion_api_secret.encode('utf-8'),
            url_ts, hashlib.sha512
        ).hexdigest()

    def _ion_request(self, endpoint: str, params: dict={}) -> Any:
        timestamp = str(arrow.utcnow().timestamp)
        headers = {
            'api-auth-time': timestamp,
            'api-auth-key': self.ion_api_key,
            'api-auth-token': self._get_signature(endpoint, params, timestamp)
        }
        response = self._ion_client.get(
            'https://ionomy.com/api/v1/' + endpoint,
            params=params,
            headers=headers
        )
        data = json.loads(response.content)
        if not data['success']:
            raise Exception(data['message'])
        return data['data']

    def markets(self) -> List[Dict[str, Any]]:
        return self._ion_request('public/markets')

    @currency_to_crypto()
    def cryptos(self) -> list:
        return self._ion_request('public/currencies')
        
    def order_book(self, market: str = "") -> dict:
        return self._ion_request(
            'public/orderbook',
            {
                'market': self._cfg(market, "market"),
                'type': 'both'
            }
        )

    def market_summaries(self) -> list:
        return self._ion_request('public/markets-summaries')

    def market_summary(self, market: str = "") -> dict:
        return self._ion_request(
            'public/market-summary',
            {'market': self._cfg(market, "market")}
        )

    def market_history(self, market: str = "") -> list:
        return self._ion_request(
            'public/market-history',
            {'market': self._cfg(market, "market")}
        )

    def limit_buy(
        self,
        amount: Union[int, float],
        price: Union[int, float],
        market: str = "",
    ) -> dict:
        params = {
            'market': self._cfg(market, "market"),
            'amount': f'{amount:.8f}',
            'price': f'{price:.8f}'
        }
        return self._ion_request('market/buy-limit', params)
    
    def limit_sell(
        self,
        amount: Union[int, float],
        price: Union[int, float],
        market: str = ""
    ) -> dict:
        params = {
            'market': self._cfg(market, "market"),
            'amount': f'{amount:.8f}',
            'price': f'{price:.8f}'
        }
        return self._ion_request('market/sell-limit', params)

    def cancel_order(self, orderId: str) -> bool:
        self._ion_request('market/cancel-order', {'orderId': orderId})
        return True

    def open_orders(self, market: str = "") -> list:
        return self._ion_request(
            'market/open-orders',
            {'market': self._cfg(market, "market")}
        )

    @currency_to_crypto()
    def balances(self) -> list:
        return self._ion_request('account/balances')

    @currency_to_crypto()
    def balance(self, currency: str = "") -> dict:
        return self._ion_request(
            'account/balance',
            {'currency': self._cfg(currency, "currency")}
        )

    @currency_to_crypto()
    def deposit_address(self, currency: str = "") -> dict:
        return self._ion_request(
            'account/deposit-address',
            {'currency': self._cfg(currency, "currency")}
        )

    @currency_to_crypto()
    def deposit_history(self, currency: str = "") -> list:
        return self._ion_request(
            'account/deposit-history',
            {'currency': self._cfg(currency, "currency")}
        )

    @currency_to_crypto()
    def withdrawal_history(self, currency: str = "") -> dict:
        return self._ion_request(
            'account/withdrawal-history',
            {'currency': self._cfg(currency, "currency")}
        )

    def get_order_status(self, orderId: str) -> dict:
        return self._ion_request('account/order', {'orderId': orderId})
