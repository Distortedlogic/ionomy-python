import random

from deap import algorithms, tools
from deap.algorithms import varAnd


class ChadArmy:
    def __init__(self, toolbox, stats):
        self.toolbox = toolbox
        self.stats = stats
    def war(self, ngen, pop_size):
        random.seed(318)

        population = self.toolbox.population(n=pop_size)
        halloffame = tools.HallOfFame(1)
        toolbox = self.toolbox
        stats = self.stats

        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields)

        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        print(logbook.stream)

        for gen in range(1, ngen + 1):
            offspring = toolbox.select(population, len(population))
            offspring = varAnd(offspring, toolbox, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Replace the current population by the offspring
            population[:] = offspring

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

        return population, logbook

        pop, log = algorithms.eaSimple(
            pop,
            self.toolbox,
            0.5,
            0.7,
            ngen,
            stats=self.stats,
            halloffame=hof,
            verbose=True
        )
        self.omega = hof[0]
