import multiprocessing

from deap import base, creator, tools
import random
import numpy as np
from pandas.core.frame import DataFrame

from .bae import Bae
from .chad import Chad
from .environment import Environment
from .model import Model

config = {
    "iterations": 25,
    "initial_capital": 10_000,
    "max_buy": 0.1,
    "max_sell": 0.1,
    "output_size": 3
}

class DemiChad:
    def __init__(self, ohlcv_df: DataFrame) -> None:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        self.default_toolbox()
        self.bae = Bae(ohlcv_df, **config, toolbox=self.toolbox, stats=self.stats)
    def default_toolbox(self):
        self.toolbox = base.Toolbox()
        self.toolbox.register("map", multiprocessing.Pool().map)
        self.toolbox.register("attr_float", random.uniform, -2, 2)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
    def meta_evolve(self):
        return self.bae.optimize()
