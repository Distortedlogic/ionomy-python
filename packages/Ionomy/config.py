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
        """Base Class - Handles configuration dictionary and defaults

        Arguments:
            crypto {str} -- [Crypto Currency. Defaults to HIVE]
            base {str} -- [Base Currency. Defaults to BTC]
            time {str} -- [Time Frame. Defaults to day]
            exchange {str} -- [Defaults to Binance]
            market {str} -- [Ionomy Market. Defaults to btc-hive]
        """
        self.config: Dict[str, str] = {
            "crypto": self._default(crypto, "crypto"),
            "base": self._default(base, "base"),
            "time": self._default(time, "time"),
            "exchange": self._default(exchange, "exchange"),
            "market": self._default(market, "market"),
        }

    def _default(self, arg: str, cfg_key: str) -> str:
        """Checks if the arg exists and returns it if so
        else it will return the corresponding entry in the class default dictionary

        Arguments:
            arg {str} -- [empty if user did not provide in higher order function]
            cfg_key {str} -- [corresponding default configuration key]

        Returns:
            str -- [arg or corresponding default dict value]
        """
        return arg if arg else self.default_config[cfg_key]

    def _cfg(self, arg: str, cfg_key: str) -> str:
        """Checks if the arg exists and returns it if so
        else it will return the corresponding entry in the class default dictionary

        Arguments:
            arg {str} -- [empty if user did not provide in higher order function]
            cfg_key {str} -- [corresponding instance configuration key]

        Returns:
            str -- [arg or corresponding instance configuration dict value]
        """
        return arg if arg else self.config[cfg_key]

    @staticmethod
    def _cfg_check(arg: str, cfg_key: str, cfg_dict: dict) -> str:
        
        return arg if arg else cfg_dict[cfg_key]