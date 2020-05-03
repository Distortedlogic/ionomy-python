# ionomy-python
Python package for working with Ionomy API (ionomy.com)

# Installation
``` pip install ionomy-python ```

# Playground
with docker:
- clone repo
- cd into repo
- make filled out copy of `example.env` as `env` in root directory
- `docker-compose up --build`
- copy link output to console with token
- insert into browser
- profit??

without docker:
- open the jupyter notebooks (.ipynb) in notebooks folder via your preferred method
- profit??

# Basic Ionomy
```
from Ionomy import Ionomy
from decouple import config

# view market_names for more market choices
HIVE_MARKET = 'btc-hive'

# view currencies for more market choices
CURRENCY = 'hive'

ionomy = Ionomy(config('IONOMY_KEY'), config('IONOMY_SECRET'))
```

`market_names = ionomy.market_names`

`markets = ionomy.markets()`

`currencies = ionomy.currencies()`

`order_book = ionomy.order_book(HIVE_MARKET)`

`market_summaries = ionomy.market_summaries()`

`hive_market_summary = ionomy.market_summary(HIVE_MARKET)`

`hive_market_history = ionomy.market_history(HIVE_MARKET)`

`balances = ionomy.balances()`

`open_orders = ionomy.open_orders(HIVE_MARKET)`

`hive_balance = ionomy.balance(CURRENCY)`

`deposit_address = ionomy.deposit_address(CURRENCY)`

`deposit_history = ionomy.deposit_history(CURRENCY)`

`withdrawal_history = ionomy.withdrawal_history(CURRENCY)`

`current_price = ionomy.get_spot_price()`


# IonPanda - Returns Pandas Dataframes when applicable plus extra methods
```
from Ionomy import IonPanda
from decouple import config

HIVE_MARKET = 'btc-hive'
CURRENCY = 'hive'

ionpd = IonPanda(config('IONOMY_KEY'), config('IONOMY_SECRET'))
```

`market_names = ionpd.market_names`

`markets_pd = ionpd.markets()`

`currencies_pd = ionpd.currencies()`

`order_book_pd = ionpd.order_book(HIVE_MARKET)`

`market_summaries_pd = ionpd.market_summaries()`

`hive_market_summary = ionpd.market_summary(HIVE_MARKET)`

`hive_market_history_pd = ionpd.market_history(HIVE_MARKET)`

`open_orders_pd = ionpd.open_orders(HIVE_MARKET)`

`balances_pd = ionpd.balances()`

`hive_balance = ionpd.balance(CURRENCY)`

`deposit_address = ionpd.deposit_address(CURRENCY)`

`deposit_history_pd = ionpd.deposit_history(CURRENCY)`

`withdrawal_history_pd = ionpd.withdrawal_history(CURRENCY)`

`current_price = ionpd.get_spot_price()`

`min_ask = ionpd.min_ask(HIVE_MARKET)`

`max_bid = ionpd.max_bid(HIVE_MARKET)`
```