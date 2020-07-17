import random, json

import numpy as np
import pandas as pd
from deap import tools

from .rules.rsi import rsi_mutate, rsi_rule, rsi_signals
from .rules.macd import macd_mutate, macd_rule, macd_signals
from .rules.ema import ema_mutate, ema_rule, ema_signals


class Environment:
    def __init__(self, df, warmup = 250):
        self.df = df
        self.warmup = warmup
        self.last = self.df['close'].iloc[-1]
        self.names = []

    def print(self, individual):
        print(json.dumps(dict(zip(self.names, individual)), indent=4))

    def attributes(self, toolbox):
        rsi_attrs, rsi_names = rsi_rule(toolbox)
        macd_attrs, macd_names = macd_rule(toolbox)
        ema_attrs, ema_names = ema_rule(toolbox)

        self.names = rsi_names + macd_names + ema_names
        self.lengths = np.cumsum([len(rsi_attrs), len(macd_attrs), len(ema_attrs)])
        return rsi_attrs + macd_attrs + ema_attrs

    def mutate(self, individual, mu, sigma, indpb):
        rsi_attrs = individual[:self.lengths[0]]
        macd_attrs = individual[self.lengths[0]:self.lengths[1]]
        ema_attrs = individual[self.lengths[1]:self.lengths[2]]

        individual[:self.lengths[0]] = rsi_mutate(rsi_attrs, mu, sigma, indpb)
        individual[self.lengths[0]:self.lengths[1]] = macd_mutate(macd_attrs, mu, sigma, indpb)
        individual[self.lengths[1]:self.lengths[2]] = ema_mutate(ema_attrs, mu, sigma, indpb)

        return individual,

    def explore(self, individual):
        rsi_attrs = individual[:self.lengths[0]]
        macd_attrs = individual[self.lengths[0]:self.lengths[1]]
        ema_attrs = individual[self.lengths[1]:self.lengths[2]]

        rsi, rsi_delay = rsi_signals(self.df, self.warmup, rsi_attrs)
        macd, macd_delay = macd_signals(self.df, self.warmup, macd_attrs)
        ema, ema_delay = ema_signals(self.df, self.warmup, ema_attrs)

        self.delay = max(rsi_delay, macd_delay, ema_delay)

        signal_sum = rsi + macd + ema

        conditions = [signal_sum <= -2, signal_sum >= 2]
        choices = [-1, 1]
        signals = pd.Series(np.select(conditions, choices, default=0), name='signal')
        signals = signals[self.delay:].reset_index(drop=True)

        state = pd.concat([signals, self.prices()], axis=1)
        return state[state != 0]

    def prices(self):
        return self.df.loc[self.warmup + self.delay:, 'close'].reset_index(drop=True)
