import multiprocessing
import operator
import random

import pandas as pd
import numpy as np
from deap import algorithms, base, creator, gp, tools
from pandas.core.frame import DataFrame

from .primitives import build_pset
from .utils.safe_gen import generate_safe


class Demichad:
    def __init__(self, df: DataFrame):
        self.df = df
        pset, terminal_types = build_pset()

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

        toolbox = base.Toolbox()
        toolbox.register("map", multiprocessing.Pool().map)
        toolbox.register("expr", generate_safe, pset=pset, min_=2, max_=5, terminal_types=terminal_types)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))

        self.pset = pset
        self.toolbox = toolbox

    def evolve(self):
        random.seed(318)

        pop = self.toolbox.population(n=300)
        hof = tools.HallOfFame(1)
        
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)
        pop, log = algorithms.eaSimple(
            pop,
            self.toolbox,
            0.5,
            0.1,
            40,
            stats=mstats,
            halloffame=hof,
            verbose=True
        )
