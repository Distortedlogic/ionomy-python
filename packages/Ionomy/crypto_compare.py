import json
import websockets
import asyncio

from typing import Any, Dict
from requests import Session

HTTP = "https://min-api.cryptocompare.com/data"
WSS = "wss://streamer.cryptocompare.com/v2?api_key="

class CryptoCompare:
    _client: Session = Session()

    def _request(self, endpoint: str, params: Dict[str, str]) -> Any:
        return self._client.get(HTTP + endpoint, params=params).json()
    
    def ohlcv(self, currency: str, base: str, time: str) -> list:
        resp = self._request(f"/v2/histo{time}", {"fsym": currency, "tsym": base})
        if resp["Response"] == "Success":
            return resp["Data"]["Data"]
        else:
            raise Exception(resp["Message"])