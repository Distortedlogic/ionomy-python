from typing import Dict

class Config(object):
    default_config: Dict[str, str] = {
        "crypto": "HIVE",
        "base": "BTC",
        "time": "day",
        "exchange": "Binance",
        "market": "btc-hive"
    }

    def __init__(
        self,
        crypto: str,
        base: str,
        time: str,
        exchange: str,
        market: str
    ) -> None:
        self.config = {
            "crypto": self._default(crypto, "crypto"),
            "base": self._default(base, "base"),
            "time": self._default(time, "time"),
            "exchange": self._default(exchange, "exchange"),
            "market": self._default(market, "market"),
        }

    def _default(self, arg: str, cfg_key: str) -> str:
        return arg if arg else self.default_config[cfg_key]

    def _cfg(self, arg: str, cfg_key: str) -> str:
        return arg if arg else self.config[cfg_key]

    @staticmethod
    def _cfg_check(arg: str, cfg_key: str, cfg_dict: dict) -> str:
        return arg if arg else cfg_dict[cfg_key]