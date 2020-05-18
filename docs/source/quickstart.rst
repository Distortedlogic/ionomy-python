Quick Start
===========

Playground
----------

Start up a jupyter lab instance with all the tutorial notebooks.

.. code:: bash

    docker run -p 8888:8888 ionomy-playground:latest
    
Copy and Paste the url with token from terminal output into your browser.

Open the tutorial notebook of interest or create your own to experiment with!

Install
-------

.. code:: bash

    pip install ionomy-python

Imports and Common Params
-------------------------

The primary classes for use are IonPanda and BitTA.

There are lower level classes if needed: Ionomy, BitTrex, and BitPanda

Ionomy is a raw API wrapper, no extras.

The same with BitTrex, which is a raw api wrapper.

BitPanda is the bittrex equivalent of IonPanda,

but without the TA methods.

.. code:: python

    from Ionomy import IonPanda, BitTA

    # Common Params
    MARKET = 'btc-hive'
    CURRENCY = 'hive'
    BASE = 'btc'
    TIME = 'day'

Instantiate
-----------

You must provide your Ionomy or BitTrex API key/secret, respectively.

IMPORTANT: I opted for the package user to manually load and update ohlcv data

This allows for control of the number of API call, since there are limits (See Exchanges Websites).

.. code:: python

    ionpd = IonPanda(IONOMY_KEY, IONOMY_SECRET)
    bta = BitTA(BITTREX_KEY, BITTREX_SECRET)

    # IMPORTANT - call this method to load/update ohlcv data
    bta.update(CURRENCY, BASE, TIME)

Common Methods
--------------

The "tutorials" provides examples of each method and their returns.

The "modules" provides the detailed code documentation.

Here, I will show one method from each classed being used by the highest order classes.

.. code:: python

    # returns a regular dictionary from the raw json
    # I saw no benefit from having a single row dataframe returned
    market_summary = ionpd.market_summary(MARKET)

    # returns a pandas dataframe
    order_book_df = ionpd.order_book(MARKET)

    # returns a regular dictionary from the raw json
    market_summary = bta.market_summary(MARKET)

    # returns a pandas dataframe
    order_book_df = bta.order_book(MARKET)

    # TA method are returned as DataFrames or Series
    rsi_series = rsi_series = bta.rsi(length=1, drift=1, offset=0)

The most common TA methods are implemented and available
Due to python limitations, such as floating point arthimetic, etc,
results will/may differ from the standard talib package
and other TA implementations on the same data.
This difference should be within a very small margin of error from a c++ implementation.