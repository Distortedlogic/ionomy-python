from Ionomy import IonPanda, BitTA
from decouple import config
import time

MARKET = 'btc-hive'
CURRENCY = 'hive'
BASE = 'btc'
TIME = 'day'

class ArbBot:
    def __init__(self):
        self.ionpd = IonPanda(config('ION_KEY'), config('ION_SECRET'))
        self.bta = BitTA(config('TREX_KEY'), config('TREX_SECRET'))
        self.bta.update(CURRENCY, BASE, TIME)
        self.current_orders_df = self.ionpd.open_orders(MARKET)
        self.current_ask = self.bta.ticker(MARKET)["Ask"]

    def run(self):
        while True:
            status_orders_df = self.ionpd.open_orders(MARKET)
            if status_orders_df == self.current_orders_df:
                pass
            else:
                pass



