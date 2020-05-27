import random, multiprocessing
from operator import attrgetter

import numpy as np
from deap import algorithms, base, creator, tools
from pandas.core.frame import DataFrame

from .model import Model
from .environment import Environment
from .chad import Chad

window_size = 32
network_size = 500

population_size = 100
tournsize = 10

mu = 0
sigma = 1.5
indpb = 0.05
cxpb = 0.3
mutpb = 0.5

initial_capital = 10_000
max_buy = 0.1
max_sell = 0.1
FEE_RATE = 0.003
output_size = 3

class DemiChad:
    def __init__(
        self,
        ohlcv_df: DataFrame
    ):
        np.random.seed(0)
        self.ohlcv_df = ohlcv_df
        self.population_size = population_size
        self.tournsize = tournsize
        self.mu = mu
        self.sigma = sigma
        self.indpb = indpb
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.history = tools.History()
        env = Environment(ohlcv_df, window_size, initial_capital, max_buy, max_sell)
        self.chad = Chad(network_size, output_size, env)
        self.build_toolbox()

    def build_toolbox(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        self.toolbox = base.Toolbox()
        self.toolbox.register("map", multiprocessing.Pool().map)
        self.toolbox.register("attr_float", random.uniform, -2, 2)
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.attr_float,
            n=len(self.chad)
        )
        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual,
            n=self.population_size
        )
        self.toolbox.register("evaluate", self.chad.fitness)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register(
            "mutate",
            tools.mutGaussian,
            mu=self.mu,
            sigma=self.sigma,
            indpb=self.indpb
        )
        self.toolbox.decorate("mate", self.history.decorator)
        self.toolbox.decorate("mutate", self.history.decorator)
        self.toolbox.register("select", tools.selTournament, tournsize=tournsize)

        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

    def evolve(self, ngen, verbose=__debug__):
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (self.stats.fields)

        # primordial ooze
        population = self.toolbox.population()
        self.history.update(population)
        self.halloffame = tools.HallOfFame(1)
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        self.halloffame.update(population)
        record = self.stats.compile(population)
        logbook.record(gen=0, nevals=len(invalid_ind), **record)

        best = []
        best_ind = max(population, key=attrgetter("fitness"))
        best.append(best_ind)
        for gen in range(1, ngen + 1):
            offspring = self.toolbox.select(population, len(population))
            offspring = map(self.toolbox.clone, offspring)
            offspring = algorithms.varAnd(offspring, self.toolbox, self.cxpb, self.mutpb)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            best_ind = max(offspring, key=attrgetter("fitness"))
            best.append(best_ind)
            self.halloffame.update(offspring)
            population[:] = offspring
            record = self.stats.compile(population)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            print(logbook.stream)
        return population, logbook, best