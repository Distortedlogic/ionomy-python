import random
import pickle

import numpy as np
from deap import algorithms, creator, tools

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
        fitness_stats,
        **kwargs
    ) -> None:
        self.population_size = population_size
        self.tournsize = tournsize
        self.mu = mu
        self.sigma = sigma
        self.indpb = indpb
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.history = tools.History()
        self.env = env
        self.chad = Chad(output_size, env)
        self.nature = self.create_nature(toolbox)
        self.fitness_stats = fitness_stats

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

    def new_time_frame(self, tf):
        self.nature.unregister("evaluate")
        self.nature.register("evaluate", self.chad.fitness, tf=tf)

    def war(self, ngen) -> float:
        try:
            cp = self.load()
            population = cp["population"]
            generation = cp["generation"]
            time_frame = cp["time_frame"]
            halloffame = cp["halloffame"]
            logbook = cp["logbook"]
            random.setstate(cp["rndstate"])
        except:
            logbook = tools.Logbook()
            logbook.header = ['gen', 'nevals'] + (self.fitness_stats.fields)
            population = self.nature.population()
            halloffame = tools.HallOfFame(1)
            time_frame = self.env.window_size
            generation = 1
        
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = self.nature.map(self.nature.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            halloffame.update(population)
            fitness_record = self.fitness_stats.compile(population)
            logbook.record(gen=0, nevals=len(invalid_ind), **fitness_record)

        for tf in range(time_frame, self.env.length - 2000, 2000):
            self.new_time_frame(tf)
            for gen in range(generation, ngen + 1):
                offspring = self.nature.select(population, len(population))
                offspring = map(self.nature.clone, offspring)
                offspring = algorithms.varAnd(offspring, self.nature, self.cxpb, self.mutpb)
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = self.nature.map(self.nature.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                halloffame.update(offspring)
                population[:] = offspring
                fitness_record = self.fitness_stats.compile(population)
                logbook.record(gen=gen, nevals=len(invalid_ind), **fitness_record)
                print(logbook.stream)
                if gen % 25 == 0:
                    cp = dict(
                        population=population,
                        generation=gen,
                        time_frame=tf,
                        halloffame=halloffame,
                        logbook=logbook,
                        rndstate=random.getstate()
                    )
                    self.checkpoint(cp)
        self.omega = halloffame[0]
        return self.omega.fitness.values[0]