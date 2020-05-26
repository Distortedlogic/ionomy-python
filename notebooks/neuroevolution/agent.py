'''Trains a convolutional neural network on sample images from the environment
using neuroevolution to maximize the ability to discriminate between input
images.

Reference:
Koutnik, Jan, Jurgen Schmidhuber, and Faustino Gomez. "Evolving deep
unsupervised convolutional networks for vision-based reinforcement
learning." Proceedings of the 2014 conference on Genetic and
evolutionary computation. ACM, 2014.
'''

import random
from operator import attrgetter

import numpy as np
from deap import algorithms, base, creator, tools
from matplotlib import pyplot as plt
from pandas.core.frame import DataFrame

from .model import Model

window_size = 32
network_size = 500
output_size = 3
tournsize = 10
mu = 0
sigma = 1.5
indpb = 0.05

FEE_RATE = 0.003
output_size = 3

class DemiGod:
    def __init__(
        self,
        ohlcv_df: DataFrame,
        window_size: int,
        network_size: int,
        population_size: int,
        tournsize: int,
        mu: float,
        sigma: float,
        indpb: float,
        cxpb: float,
        mutpb: float
    ):
        self.ohlcv_df = ohlcv_df
        self.window_size = window_size
        self.network_size = network_size
        self.output_size = output_size
        self.population_size =population_size
        self.tournsize = tournsize
        self.mu = mu
        self.sigma = sigma
        self.indpb = indpb
        self.model = Model(window_size, network_size, output_size)
        self.cxpb = cxpb
        self.mutpb = mutpb
        np.random.seed(0)
        self.history = tools.History()
        self.toolbox = self.build_toolbox()

    def fitness(self, individual):
        self.model.set_weights(np.asarray(individual))
        fitness = 0
        return fitness,

    def build_toolbox(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.uniform, -1, 1)
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.attr_float,
            n=len(self.model.flatten())
        )
        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual,
            n=self.population_size
        )
        self.toolbox.register("evaluate", self.fitness)
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
        self.population = self.toolbox.population()
        self.history.update(self.population)
        self.halloffame = tools.HallOfFame(1)

        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (self.stats.fields)

        # primordial ooze
        population = self.toolbox.population()
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