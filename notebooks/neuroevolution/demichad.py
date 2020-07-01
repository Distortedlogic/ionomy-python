import multiprocessing

from deap import base, creator, tools
import random
import numpy as np
from pandas.core.frame import DataFrame

from .bae import Bae
from .chad import Chad
from .environment import Environment
from .chad_army import ChadArmy

env_config = {
    "initial_capital": 10_000,
    "max_buy": 0.1,
    "max_sell": 0.1
}

defaults = {
    "window_size": 32,
    "network_size": 400,
    "population_size": 100,
    "tournsize": 15,
    "mu": -0.25,
    "sigma": 0.65,
    "indpb": 0.05,
    "cxpb": 0.6,
    "mutpb": 0.7,
    "output_size": 3,
    "iterations": 25
}

search_grid = {
    "mu": (-0.25, 0.25),
    'sigma': (0.5, 0.99),
    "indpb": (0.05, 0.20),
    "cxpb": (0.25, 0.75),
    "mutpb": (0.25, 0.8)
}

opt_config = {
    "init_points": 30,
    "n_iter": 50,
    "acq": 'ei',
    "xi": 0.0
}

class DemiChad:
    def __init__(self, ohlcv_df: DataFrame) -> None:
        self.ohlcv_df = ohlcv_df
        creator.create("FitnessMulti", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        self.create_tools()
        self.bae = Bae(
            ohlcv_df,
            env_config = env_config,
            tools = self.tools,
            defaults = defaults,
            search_grid = search_grid
        )
    def create_tools(self):
        toolbox = base.Toolbox()
        toolbox.register("map", multiprocessing.Pool().map)
        toolbox.register("attr_float", random.uniform, -2, 2)
        toolbox.register("mate", tools.cxTwoPoint)
        fitness_stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        fitness_stats.register("f_avg", np.mean)
        fitness_stats.register("f_max", np.max)
        self.tools = {
            'toolbox': toolbox,
            'fitness_stats': fitness_stats
        }
    def meta_evolve(self):
        self.bae.optimize(**opt_config)
    def evolve(self, ngen):
        params = self.bae.params if self.bae.params else defaults
        self.env = Environment(self.ohlcv_df, **env_config, window_size=params["window_size"])
        self.army = ChadArmy(**params, **self.tools, env=self.env)
        return self.army.war(ngen)
    def omega(self):
        self.omega_chad = Chad(
            defaults["output_size"],
            env=self.env
        )
        self.omega_chad.fitness(self.army.omega, tf=None)
        self.omega_chad.brain.model.save('./chad')
    def plot(self):
        self.omega_chad.plot()