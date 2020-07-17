import random
from deap import tools

import numpy as np
import pandas as pd

def macd_rule(toolbox):
    macd_names = [
        'macd_fast',
        'macd_slow',
        'macd_signal',
        'macd_offset',
        'macd_delay',
        'lower_threshold',
        'upper_threshold'
    ]

    toolbox.register("macd_fast", random.randint, 1, 50)
    toolbox.register("macd_slow", random.randint, 1, 100)
    toolbox.register("macd_signal", random.randint, 1, 50)
    toolbox.register("macd_offset", random.randint, 1, 10)
    toolbox.register("macd_delay", random.randint, 1, 10)
    toolbox.register("lower_threshold", random.uniform, -1, 0)
    toolbox.register("upper_threshold", random.uniform, 0, 1)

    return (
        toolbox.macd_fast,
        toolbox.macd_slow,
        toolbox.macd_signal,
        toolbox.macd_offset,
        toolbox.macd_delay,
        toolbox.lower_threshold,
        toolbox.upper_threshold
    ), macd_names

def macd_mutate(ema_attrs, mu, sigma, indpb):
    fast = tools.mutUniformInt([ema_attrs[0]], 1, 50, indpb)[0][0]
    slow = tools.mutUniformInt([ema_attrs[1]], fast + 1, 100, indpb)[0][0]
    signal = tools.mutUniformInt([ema_attrs[2]], 1, 50, indpb)[0][0]
    offset = tools.mutUniformInt([ema_attrs[3]], 0, 10, indpb)[0][0]
    delay = tools.mutUniformInt([ema_attrs[4]], 0, 10, indpb)[0][0]
    lower = tools.mutGaussian([ema_attrs[5]], mu, sigma, indpb)[0][0]
    upper = tools.mutGaussian([ema_attrs[6]], mu, sigma, indpb)[0][0]
    return [fast, slow, signal, offset, delay, lower, upper]

def macd_signals(df, warmup, macd_attrs):
    fast = macd_attrs[0]
    slow = macd_attrs[1]
    signal = macd_attrs[2]
    offset = macd_attrs[3]
    delay = macd_attrs[4]
    lower = macd_attrs[5]
    upper = macd_attrs[6]
    macd_full = df.ta.macd(fast=fast, slow=slow, signal=signal, offset=offset)
    macd_full = macd_full[warmup:].reset_index(drop=True)
    macdh = macd_full.iloc[:, 1]

    conditions = [macdh < lower, macdh > upper]
    choices = [1, -1]
    triggers = pd.Series(np.select(conditions, choices, default=0), name='triggers')

    accum_signal = triggers.rolling(delay).sum()

    conditions = [accum_signal == delay, accum_signal == -delay]
    choices = [1, -1]
    signals = pd.Series(np.select(conditions, choices, default=0), name='signal')

    return signals, delay