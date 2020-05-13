import arrow
import hashlib
import hmac

from furl import furl

from typing import Any, Dict, List, Optional, Union
from requests import Session

HTTP = " https://api.bittrex.com/api/v1.1"
headers = {"content-type": "application/json"}

class BitTrex:
    def __init__(self, api_key: str, secret_key: str) -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self._client: Session = Session()
        self._params = {"apikey": self.api_key}

    def _get_signature(self, endpoint: str, params: Optional[Dict[str, str]]) -> str:
        api_furl = furl(HTTP + endpoint)
        api_furl.args = params
        return hmac.new(
            self.secret_key.encode('utf-8'),
            api_furl.url.encode('utf-8'),
            hashlib.sha512
        ).hexdigest()

    def _request(self, endpoint: str, params: Optional[Dict[str, str]] = {}) -> Any:
        self._params["nonce"] = str(arrow.utcnow().timestamp)
        headers["apisign"] = self._get_signature(endpoint, params)
        resp = self._client.get(HTTP + endpoint, params=params, headers=headers).json()
        if not resp["success"]:
            raise Exception(resp["message"])
        return resp["result"]

    def markets(self) -> List[Dict[str, Union[bool, str, float]]]:
        return self._request("/public/getmarkets")

    def currencies(self) -> List[Dict[str, Union[bool, str, float, None]]]:
        return self._request("/public/getcurrencies")
    
    def ticker(self, market: str) -> Dict[str, float]:
        return self._request("/public/getticker", {"market": market})

    def market_summaries(self) -> List[Dict[str, Union[bool, str, int, float, None]]]:
        return self._request("/public/getmarketsummaries")

    def market_summary(self, market: str):
        return self._request("/public/getmarketsummary", {"market": market})

    def order_book(self, market: str):
        return self._request("/public/getorderbook", {"market": market, "type": "both"})

    def market_history(self, market: str):
        return self._request("/public/getmarkethistory", {"market": market})

    def buy_limit(
        self,
        market: str,
        quantity: float,
        rate: float,
        timeInForce: str
    ) -> str:
        params = {
            "market": market,
            "quantity": quantity,
            "rate": rate,
            "timeInForce": timeInForce
        }
        return self._request("/market/buylimit", params)["uuid"]

    def sell_limit(
        self,
        market: str,
        quantity: float,
        rate: float,
        timeInForce: Optional[str]
    ) -> str:
        params = {
            "market": market,
            "quantity": quantity,
            "rate": rate,
            "timeInForce": timeInForce
        }
        return self._request("/market/selllimit", params)["uuid"]

    def cancel(self, uuid: str) -> str:
        return self._request("/market/cancel", {"uuid": uuid})["uuid"]

    def open_orders(self, market: str):
        return self._request("/market/getopenorders", {"market": market})

    def balances(self):
        return self._request("/account/getbalances")

    def balance(self, currency: str):
        return self._request("/account/getbalance", {"currency": currency})

    def deposit_address(self, currency: str):
        return self._request("/account/getdepositaddress", {"currency": currency})

    def withdraw(self, currency: str, quantity: float, address: str, paymentid: Optional[str]):
        params = {
            "currency": currency,
            "quantity": quantity,
            "address": address
        }
        if paymentid:
            params["paymentid"] = paymentid
        return self._request("/account/withdraw", params)

    def get_order(self, uuid: str):
        return self._request("/account/getorder", {"uuid": uuid})

    def order_history(self):
        return self._request("/account/getorderhistory")

    def withdrawal_history(self, currency: Optional[str]):
        params = {}
        if currency:
            params["currency"] = currency
        return self._request("/account/getwithdrawalhistory", params)

    def deposit_history(self, currency: Optional[str]):
        params = {}
        if currency:
            params["currency"] = currency
        return self._request("/account/getdeposithistory", params)
