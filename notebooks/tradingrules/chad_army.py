import pickle
import random
import multiprocessing

import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from deap import algorithms, base, creator, tools

from .environment import Environment
from .chad import Chad

class ChadArmy:
    def __init__(
        self,
        population_size: int,
        tournsize: int,
        mu: float,
        sigma: float,
        indpb: float,
        cxpb: float,
        mutpb: float,
        env: Environment,
        toolbox,
        fitness_stats
    ):
        self.population_size = population_size
        self.tournsize = tournsize
        self.mu = mu
        self.sigma = sigma
        self.indpb = indpb
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.history = tools.History()
        self.env = env
        self.chad = Chad(env)
        self.update_toolbox(toolbox)
        self.fitness_stats = fitness_stats

    def update_toolbox(self, toolbox):
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
            tools.initCycle,
            creator.Individual,
            self.env.attributes(toolbox),
            n=1
        )
        toolbox.register(
            "population",
            tools.initRepeat,
            list,
            toolbox.individual,
            n=self.population_size
        )
        toolbox.register(
            "mutate",
            self.env.mutate,
            mu = self.mu,
            sigma = self.sigma,
            indpb = self.indpb
        )
        toolbox.register("mate", tools.cxUniform, indpb = self.indpb)
        toolbox.register("select", tools.selTournament, tournsize = self.tournsize)

        toolbox.register("evaluate", self.chad.fitness)
        self.toolbox = toolbox

    def war(self, ngen):
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (self.fitness_stats.fields)
        population = self.toolbox.population()
        halloffame = tools.HallOfFame(1)
        generation = 1

        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        halloffame.update(population)
        self.history.update(population)
        fitness_record = self.fitness_stats.compile(population)
        logbook.record(gen=0, nevals=len(invalid_ind), **fitness_record)
        print(logbook.stream)

        for gen in range(generation, ngen + 1):
            offspring = self.toolbox.select(population, len(population))
            offspring = map(self.toolbox.clone, offspring)
            offspring = algorithms.varAnd(offspring, self.toolbox, self.cxpb, self.mutpb)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            halloffame.update(offspring)
            self.history.update(population)
            population[:] = offspring
            fitness_record = self.fitness_stats.compile(population)
            logbook.record(gen=gen, nevals=len(invalid_ind), **fitness_record)
            print(logbook.stream)
            if gen % 25 == 0:
                cp = dict(
                    population=population,
                    generation=gen,
                    halloffame=halloffame,
                    logbook=logbook,
                    rndstate=random.getstate()
                )
                self.checkpoint(cp)
        self.omega = halloffame[0]
        return self.omega.fitness.values[0]

    def load(self):
        with open("checkpoint.pkl", "rb") as cp_file:
            cp = pickle.load(cp_file)
        return cp

    def checkpoint(self, cp):
        with open("checkpoint.pkl", "wb") as cp_file:
            pickle.dump(cp, cp_file)