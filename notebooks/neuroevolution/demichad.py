import multiprocessing

from deap import base, creator, tools
import random
import numpy as np
from pandas.core.frame import DataFrame

from .bae import Bae
from .chad import Chad
from .environment import Environment
from .model import Model
from .chad_army import ChadArmy

env_config = {
    "initial_capital": 10_000,
    "max_buy": 0.1,
    "max_sell": 0.1
}

chad_config = {
    "window_size": 32,
    "network_size": 500,
    "population_size": 100,
    "tournsize": 25,
    "mu": 0,
    "sigma": 0.65,
    "indpb": 0.15,
    "cxpb": 0.3,
    "mutpb": 0.5,
    "output_size": 3
}

bae_config = {
    "iterations": 25,
    "output_size": 3
}

class DemiChad:
    def __init__(self, ohlcv_df: DataFrame) -> None:
        self.ohlcv_df = ohlcv_df
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        self.default_toolbox()
        # self.bae = Bae(ohlcv_df, **env_config, **bae_config, toolbox=self.toolbox, stats=self.stats)
        self.params = {}
    def default_toolbox(self):
        self.toolbox = base.Toolbox()
        self.toolbox.register("map", multiprocessing.Pool().map)
        self.toolbox.register("attr_float", random.uniform, -2, 2)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.fitness_stats = tools.Statistics(lambda ind: ind.fitness.values[1])
        self.fitness_stats.register("f_avg", np.mean)
        # self.fitness_stats.register("f_std", np.std)
        # self.fitness_stats.register("f_min", np.min)
        self.fitness_stats.register("f_max", np.max)
        self.lifespan_stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        self.lifespan_stats.register("l_avg", np.mean)
        # self.lifespan_stats.register("l_std", np.std)
        # self.lifespan_stats.register("l_min", np.min)
        self.lifespan_stats.register("l_max", np.max)
    # def meta_evolve(self):
    #     return self.bae.optimize()
    def evolve(self, ngen):
        params = self.params if self.params else chad_config
        self.env = Environment(self.ohlcv_df, **env_config, window_size=params["window_size"])
        army_params = {
            "lifespan_stats": self.lifespan_stats,
            "fitness_stats": self.fitness_stats
        }
        self.army = ChadArmy(**params, toolbox=self.toolbox, **army_params, env=self.env)
        return self.army.war(ngen)
    def omega(self):
        self.omega_chad = Chad(
            chad_config["network_size"],
            chad_config["output_size"],
            env=self.env
        )
        self.omega_chad.fitness(self.army.omega)
    def plot(self):
        self.omega_chad.plot(self.omega_chad.buy_history, self.omega_chad.sell_history, self.omega_chad.results)