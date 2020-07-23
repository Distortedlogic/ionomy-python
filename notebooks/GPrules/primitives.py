from pandas.core.frame import DataFrame
import pandas

from deap import gp

from .operators import add_operators

from .rules.price import add_price_rule, price_terminals
from .rules.ema import add_ema_rule, ema_terminals
from .rules.rsi import add_rsi_rule, rsi_terminals
from .rules.filter import add_filter_rule, filter_terminals
from .rules.ret_types import signals

def build_pset():
    pset = gp.PrimitiveSetTyped("MAIN", [DataFrame], signals)
    pset.renameArguments(ARG0="df")

    add_operators(pset)
    add_price_rule(pset)
    add_rsi_rule(pset)
    add_ema_rule(pset)
    add_filter_rule(pset)

    terminal_types = [
        pandas.core.frame.DataFrame
    ] + filter_terminals + rsi_terminals + ema_terminals + price_terminals

    return pset, terminal_types

