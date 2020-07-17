import random
from deap import tools

import numpy as np
import pandas as pd

def rsi_rule(toolbox):
    rsi_names = [
        'rsi_length',
        'rsi_drift',
        'rsi_offset',
        'rsi_delay',
        'rsi_lower',
        'rsi_upper'
    ]
    toolbox.register("rsi_length", random.randint, 1, 200)
    toolbox.register("rsi_drift", random.randint, 0, 50)
    toolbox.register("rsi_offset", random.randint, 0, 10)
    toolbox.register("rsi_delay", random.randint, 0, 10)
    toolbox.register("rsi_lower", random.uniform, 0.0, 0.4)
    toolbox.register("rsi_upper", random.uniform, 0.6, 1)

    return (
        toolbox.rsi_length,
        toolbox.rsi_drift,
        toolbox.rsi_offset,
        toolbox.rsi_delay,
        toolbox.rsi_lower,
        toolbox.rsi_upper
    ), rsi_names

def rsi_mutate(rsi_attrs, mu, sigma, indpb):
    return [
        tools.mutUniformInt([rsi_attrs[0]], 1, 200, indpb)[0][0],
        tools.mutUniformInt([rsi_attrs[1]], 0, 50, indpb)[0][0],
        tools.mutUniformInt([rsi_attrs[2]], 0, 10, indpb)[0][0],
        tools.mutUniformInt([rsi_attrs[3]], 0, 10, indpb)[0][0],
        tools.mutGaussian([rsi_attrs[4]], mu, sigma, indpb)[0][0],
        tools.mutGaussian([rsi_attrs[5]], mu, sigma, indpb)[0][0]
    ]

def rsi_signals(df, warmup, rsi_attrs):
    length = rsi_attrs[0]
    drift = rsi_attrs[1]
    offset = rsi_attrs[2]
    delay = rsi_attrs[3]
    lower = rsi_attrs[4]
    upper = rsi_attrs[5]

    rsi = df.ta.rsi(length = length, drift = drift, offset = offset)/100
    rsi = rsi[warmup:].reset_index(drop=True)
    conditions = [rsi < lower, rsi > upper]
    choices = [1, -1]
    triggers = pd.Series(np.select(conditions, choices, default=0))

    accum_signal = triggers.rolling(delay).sum()

    conditions = [accum_signal == delay, accum_signal == -delay]
    choices = [1, -1]
    signals = pd.Series(np.select(conditions, choices, default=0), name='signal')
    return signals, delay
