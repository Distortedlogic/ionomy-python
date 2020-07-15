from multiprocessing import Pool

from deap import base, creator, tools
import random
import numpy as np
from pandas.core.frame import DataFrame

class Trader:
    def __init__(self, ohlcv_df: DataFrame) -> None:
        self.ohlcv_df = ohlcv_df
        creator.create("FitnessMulti", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        self.create_tools()
    def create_tools(self):
        toolbox = base.Toolbox()
        toolbox.register("map", Pool().map)
        toolbox.register("attr_int_1", random.randint, -2, 2)
        toolbox.register("attr_int_2", random.randint, -2, 2)
        toolbox.register("attr_int_3", random.randint, -2, 2)
        toolbox.register("attr_float_1", random.uniform, 0, 1)
        toolbox.register("mate", tools.cxTwoPoint)
        fitness_stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        fitness_stats.register("f_avg", np.mean)
        fitness_stats.register("f_max", np.max)
        self.tools = {
            'toolbox': toolbox,
            'fitness_stats': fitness_stats
        }

    def create_nature(self, toolbox):
        try:
            toolbox.unregister("individual")
            toolbox.unregister("population")
            toolbox.unregister("evaluate")
            toolbox.unregister("mutate")
            toolbox.unregister("select")
        except:
            pass
        np.random.seed(42)
        random.seed(42)
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.attr_float,
            n=len(self.chad)
        )
        toolbox.register(
            "population",
            tools.initRepeat,
            list,
            toolbox.individual,
            n=self.population_size
        )
        toolbox.register("evaluate", self.chad.fitness, tf=self.env.window_size)
        toolbox.register(
            "mutate",
            tools.mutGaussian,
            mu=self.mu,
            sigma=self.sigma,
            indpb=self.indpb
        )
        toolbox.register("select", tools.selTournament, tournsize=self.tournsize)
        return toolbox
        
    def load(self):
        with open("checkpoint.pkl", "rb") as cp_file:
            cp = pickle.load(cp_file)
        return cp

    def checkpoint(self, cp):
        with open("checkpoint.pkl", "wb") as cp_file:
            pickle.dump(cp, cp_file)