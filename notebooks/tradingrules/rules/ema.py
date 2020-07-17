import random
from deap import tools

import numpy as np
import pandas as pd

def ema_rule(toolbox):
    ema_names = [
        'ema_length',
        'ema_offset',
        'ema_delay',
        'ema_lower',
        'ema_upper'
    ]

    toolbox.register("ema_length", random.randint, 1, 200)
    toolbox.register("ema_offset", random.randint, 1, 10)
    toolbox.register("ema_delay", random.randint, 1, 10)
    toolbox.register("ema_lower", random.uniform, -1, 0)
    toolbox.register("ema_upper", random.uniform, 0, 1)

    return (
        toolbox.ema_length,
        toolbox.ema_offset,
        toolbox.ema_delay,
        toolbox.ema_lower,
        toolbox.ema_upper
    ), ema_names

def ema_mutate(ema_attrs, mu, sigma, indpb):
    return [
        tools.mutUniformInt([ema_attrs[0]], 1, 200, indpb)[0][0],
        tools.mutUniformInt([ema_attrs[1]], 1, 10, indpb)[0][0],
        tools.mutUniformInt([ema_attrs[2]], 1, 10, indpb)[0][0],
        tools.mutGaussian([ema_attrs[3]], mu, sigma, indpb)[0][0],
        tools.mutGaussian([ema_attrs[4]], mu, sigma, indpb)[0][0]
    ]

def ema_signals(df, warmup, ema_attrs):
    length = ema_attrs[0]
    offset = ema_attrs[1]
    delay = ema_attrs[2]
    lower = ema_attrs[3]
    upper = ema_attrs[4]

    ema = df.ta.ema(length = length, offset = offset)
    percent_diff = df['close'].subtract(ema).divide(df['close'])
    percent_diff = percent_diff[warmup:].reset_index(drop=True)

    conditions = [percent_diff < lower, percent_diff > upper]
    choices = [1, -1]
    triggers = pd.Series(np.select(conditions, choices, default=0))

    accum_signal = triggers.rolling(delay).sum()

    conditions = [accum_signal == delay, accum_signal == -delay]
    choices = [1, -1]
    signals = pd.Series(np.select(conditions, choices, default=0), name='signal')
    return signals, delay