from os import stat
import pickle
import random

from deap import tools, gp

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

class ChadArmy:
    def __init__(self, toolbox, stats):
        self.toolbox = toolbox
        self.stats = stats
    def log(self, pop, nevals, cast):
        record = self.stats.compile(pop)
        self.logbook.record(gen=self.gen, nevals=nevals, cast=cast, **record)
        print(self.logbook.stream)
    def evaluate_pop(self, pop, cast):
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        self.halloffame.update(pop)
        self.log(pop, len(invalid_ind), cast)

    def init_pop(self, pop_size):
        random.seed(42)

        self.logbook = tools.Logbook()
        self.logbook.header = ['gen', 'nevals', 'cast'] + (self.stats.fields)
        self.halloffame = tools.HallOfFame(1)

        pop = self.toolbox.population(n=pop_size)
        self.evaluate_pop(pop, "Init")

        cp = dict(
            pop=pop,
            halloffame=self.halloffame,
            logbook=self.logbook,
            rndstate=random.getstate()
        )
        self.checkpoint(cp, "init_pop.pkl")
        return pop
    def war(self, ngen, pop_size):
        try:
            self.gen = 0
            cp = self.load("init_pop.pkl")
            population = cp["pop"]
            if len(population) != pop_size:
                raise Exception
            self.halloffame = cp["halloffame"]
            self.logbook = cp["logbook"]
            self.logbook.header = ['gen', 'nevals', 'cast'] + (self.stats.fields)
            random.setstate(cp["rndstate"])
        except Exception as e:
            print(e)
            population = self.init_pop(pop_size)

        for self.gen in range(1, ngen + 1):
            
            clones = [self.toolbox.clone(ind) for ind in population]
            pop = self.apply_func(clones, len(clones) // 10, self.uniform_mut, "Uniform")

            for _ in range(5):
                clones = [self.toolbox.clone(ind) for ind in pop]
                mutated = self.apply_func(clones, False, self.ephem_mut, "Ephem")
                pop = self.best_of_two(mutated, pop)
                self.log(pop, 0, "best_of_two")
            self.graph(self.halloffame[0])

            for _ in range(5):
                clones = [self.toolbox.clone(ind) for ind in pop]
                mutated = self.apply_func(clones, False, self.node_mut, "Node")
                pop = self.best_of_two(mutated, pop)
                self.log(pop, 0, "best_of_two")
            self.graph(self.halloffame[0])

            for _ in range(5):
                clones = [self.toolbox.clone(ind) for ind in pop]
                mutated = self.apply_func(clones, False, self.insert_mut, "Insert")
                pop = self.best_of_two(mutated, pop)
                self.log(pop, 0, "best_of_two")
            self.graph(self.halloffame[0])

            for _ in range(5):
                clones = [self.toolbox.clone(ind) for ind in pop]
                mated = self.apply_func(clones, False, self.shrink_mut, "Shrink")
                pop = self.best_of_two(mated, pop)
                self.log(pop, 0, "best_of_two")
            self.graph(self.halloffame[0])

            for _ in range(5):
                clones = [self.toolbox.clone(ind) for ind in pop]
                mated = self.apply_func(clones, False, self.mate, "Mate")
                pop = self.best_of_two(mated, pop)
                self.log(pop, 0, "best_of_two")
            self.graph(self.halloffame[0])

            population = self.toolbox.selTournament(pop, len(pop), tournsize=5)

            if self.gen % 5 == 0:
                cp = dict(
                    pop=pop,
                    halloffame=self.halloffame,
                    logbook=self.logbook,
                    gen=self.gen,
                    rndstate=random.getstate()
                )
                self.checkpoint(cp, f"checkpoints/pop_{self.gen}.pkl")
        self.omega = self.halloffame[0]
    
    @staticmethod
    def better(competitors):
        ind1, ind2 = competitors
        if ind1.fitness.values[0] > ind2.fitness.values[0]:
            return ind1
        else:
            return ind2

    def best_of_two(self, pop1, pop2):
        return self.toolbox.map(self.better, zip(pop1, pop2))
        # best = []
        # for i in range(len(pop1)):
        #     if pop1[i].fitness.values[0] > pop2[i].fitness.values[0]:
        #         best.append(pop1[i])
        #     else:
        #         best.append(pop2[i])
        # return best

    def apply_func(self, pop, keep_best, func, cast):
        if keep_best:
            best = self.toolbox.selBest(pop, keep_best)
            offspring = func(pop)
            self.evaluate_pop(offspring, cast)
            return self.toolbox.selBest(offspring, len(offspring) - keep_best)[:] + best[:]
        else:
            offspring = func(pop)
            self.evaluate_pop(offspring, cast)
            return offspring

    def graph(self, ind=None):
        if not ind:
            ind = self.omega
        plt.rcParams["figure.figsize"] = (45, 40)

        nodes, edges, labels = gp.graph(ind)
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        pos = graphviz_layout(g)

        nx.draw_networkx_nodes(g, pos, node_size=20000, node_color='grey')
        nx.draw_networkx_edges(g, pos)
        nx.draw_networkx_labels(g, pos, labels, font_size=50)
        plt.savefig('real_test_run_1.png')
        plt.show()

    def load(self, path):
        with open(path, "rb") as cp_file:
            cp = pickle.load(cp_file)
        return cp

    def checkpoint(self, cp, path):
        with open(path, "wb") as cp_file:
            pickle.dump(cp, cp_file)

    def insert_mut(self, pop):
        for i in range(len(pop)):
            pop[i], = self.toolbox.mutInsert(pop[i])
            del pop[i].fitness.values
        return pop
    
    def shrink_mut(self, pop):
        for i in range(len(pop)):
            pop[i], = self.toolbox.mutShrink(pop[i])
            del pop[i].fitness.values
        return pop

    def uniform_mut(self, pop):
        for i in range(len(pop)):
            pop[i], = self.toolbox.mutUniform(pop[i])
            del pop[i].fitness.values
        return pop

    def node_mut(self, pop):
        for i in range(len(pop)):
            pop[i], = self.toolbox.mutNodeReplacement(pop[i])
            del pop[i].fitness.values
        return pop

    def ephem_mut(self, pop):
        for i in range(len(pop)):
            if random.random() < 0.9:
                pop[i], = self.toolbox.mutEphemeral(pop[i], mode="one")
            else:
                pop[i], = self.toolbox.mutEphemeral_rand(pop[i])
            del pop[i].fitness.values
        return pop
    
    def mate(self, pop):
        for i in range(1, len(pop), 2):
            pop[i - 1], pop[i] = self.toolbox.mate(
                pop[i - 1],
                pop[i]
            )
            del pop[i - 1].fitness.values, pop[i].fitness.values
        return pop