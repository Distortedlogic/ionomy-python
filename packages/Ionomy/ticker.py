import json
import websockets
import asyncio

from typing import Any, Dict, List, Optional, Union, overload
from requests import Session

from .config import Config

CC_HTTP = "https://min-api.cryptocompare.com/data"
CC_WSS = "wss://streamer.cryptocompare.com/v2?api_key="

class Ticker(Config):
    _cc_client: Session = Session()

    def __init__(self, cc_api_key: str = "", **kwargs) -> None:
        Config.__init__(self, **kwargs)
        self.cc_api_key = cc_api_key

    def _cc_request(self, endpoint: str, params: Dict[str, str]) -> Any:
        return self._cc_client.get(CC_HTTP + endpoint, params=params).json()

    def _params(self, crypto: str, base: str, tsyms: bool = True) -> Dict[str, str]:
        return {
            "fsym": self._cfg(crypto, "crypto"),
            "tsyms" if tsyms else "tsym": self._cfg(base, "base")
        }
    
    def spot_price(self, crypto: str = "", base: str = "") -> float:
        return self._cc_request("/price", self._params(crypto, base))[base]
    
    def ohlcv(
        self,
        crypto: str = "",
        base: str = "",
        time: str = ""
    ) -> list:
        resp = self._cc_request(
            f"/v2/histo{self._cfg(time, 'time')}",
            self._params(crypto, base, tsyms = False)
        )
        if resp["Response"] == "Success":
            return resp["Data"]["Data"]
        else:
            raise Exception(resp["Message"])

    async def _stream(self, exchange: str, crypto: str, base: str):
        if not self.cc_api_key:
            raise Exception(f'Streamer requires a crypto compare api key at {self.__name__} initialization')
        request = json.dumps({
            "action": "SubAdd",
            "subs": [f"8~{exchange}~{crypto}~{base}"],
        })
        async with websockets.connect(CC_WSS + self.cc_api_key) as websocket:
            await websocket.send(request)
            while True:
                try:
                    data = await websocket.recv()
                except websockets.ConnectionClosed:
                    break
                try:
                    data = json.loads(data)
                    return data
                except ValueError:
                    print(data)

    def streamer(self, exchange: str = "", crypto: str = "", base: str = ""):
        return asyncio.get_event_loop().run_until_complete(
            self._stream(
                self._cfg(exchange, "exchange"),
                self._cfg(crypto, "crypto"),
                self._cfg(base, "base")
            )
        )