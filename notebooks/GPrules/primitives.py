from pandas.core.frame import DataFrame
import pandas

from deap import gp

from .operators import add_operators, operator_terminals

from .rules.price import add_price_rule, price_terminals
from .rules.ema import add_ema_rule, ema_terminals
from .rules.rsi import add_rsi_rule, rsi_terminals
from .rules.filter import add_filter_rule, filter_terminals
from .rules.ret_types import signals
from .rules.cossover import add_crossover_rule, macd_terminals
from .rules.slope import add_slope_rule, slope_terminals

def build_pset():
    pset = gp.PrimitiveSetTyped("MAIN", [DataFrame], signals)
    pset.renameArguments(ARG0="ohlcv")

    add_operators(pset)
    add_price_rule(pset)
    add_rsi_rule(pset)
    add_ema_rule(pset)
    add_filter_rule(pset)
    add_crossover_rule(pset)
    add_slope_rule(pset)

    terminal_types = (
        operator_terminals +
        filter_terminals +
        rsi_terminals +
        ema_terminals +
        price_terminals +
        macd_terminals +
        slope_terminals
        )

    return pset, terminal_types

