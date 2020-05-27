import random, multiprocessing
from operator import attrgetter

import numpy as np
from deap import algorithms, base, creator, tools
from pandas.core.frame import DataFrame

from .model import Model
from .environment import Environment
from .chad import Chad

class ChadArmy:
    def __init__(
        self,
        window_size: int,
        network_size: int,
        population_size: int,
        tournsize: int,
        mu: float,
        sigma: float,
        indpb: float,
        cxpb: float,
        mutpb: float,
        env: Environment,
        output_size: int,
        toolbox,
        stats
    ) -> None:
        self.population_size = population_size
        self.tournsize = tournsize
        self.mu = mu
        self.sigma = sigma
        self.indpb = indpb
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.history = tools.History()
        self.chad = Chad(network_size, output_size, env)
        self.nature = self.create_nature(toolbox)
        self.stats = stats

    def create_nature(self, toolbox):
        try:
            toolbox.unregister("individual")
            toolbox.unregister("population")
            toolbox.unregister("evaluate")
            toolbox.unregister("mutate")
            toolbox.unregister("select")
        except:
            pass
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
        toolbox.register("evaluate", self.chad.fitness)
        toolbox.register(
            "mutate",
            tools.mutGaussian,
            mu=self.mu,
            sigma=self.sigma,
            indpb=self.indpb
        )
        toolbox.register("select", tools.selTournament, tournsize=self.tournsize)
        return toolbox
        

    def war(self, ngen) -> float:
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (self.stats.fields)

        # primordial ooze
        population = self.nature.population()
        self.halloffame = tools.HallOfFame(1)
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = self.nature.map(self.nature.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        self.halloffame.update(population)
        record = self.stats.compile(population)
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        biggest_dick = record["max"]

        for gen in range(1, ngen + 1):
            offspring = self.nature.select(population, len(population))
            offspring = map(self.nature.clone, offspring)
            offspring = algorithms.varAnd(offspring, self.nature, self.cxpb, self.mutpb)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.nature.map(self.nature.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            self.halloffame.update(offspring)
            population[:] = offspring
            record = self.stats.compile(population)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if record["max"] > biggest_dick:
                biggest_dick = record["max"]
            print(logbook.stream)
        return biggest_dick