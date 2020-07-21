import multiprocessing
import operator
import random

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, gp, tools
from deap.gp import Ephemeral
from pandas.core.frame import DataFrame

from .chad import Chad
from .primitives import build_pset
from .utils.safe_gen import generate_safe


def mutEphemeral(individual, indpb):
    ephemerals_idx = [
        index for index, node in enumerate(individual) if isinstance(node, Ephemeral)
    ]

    if len(ephemerals_idx) > 0:
        is_int_arr = [isinstance(type(individual[i])(), int) for i in ephemerals_idx]
        for i, is_int in zip(ephemerals_idx, is_int_arr):
            if is_int:
                individual[i] = type(individual[i])()

    return individual,

def mutate(individual, expr, pset):
    choice = random.randint(0, 1)
    if choice == 0:
        return gp.mutUniform(individual, expr=expr, pset=pset)
    else:
        return gp.mutEphemeral(individual, mode='one')

class DemiChad:
    def __init__(self, df: DataFrame):
        self.df = df
        pset, terminal_types = build_pset()

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

        toolbox = base.Toolbox()
        toolbox.register("map", multiprocessing.Pool().map)
        toolbox.register("expr", generate_safe, pset=pset, min_=3, max_=5, terminal_types=terminal_types)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        toolbox.register("select", tools.selTournament, tournsize=25)
        toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
        toolbox.register("mutate", mutate, expr=toolbox.expr, pset=pset)

        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        self.stats = stats

        self.chad = Chad(pset, df)
        toolbox.register("evaluate", self.chad.fitness)
        self.pset = pset
        self.toolbox = toolbox

    def evolve(self, pop_size, ngen):
        random.seed(318)

        pop = self.toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)

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
        self.chad.fitness(self.omega)

    def plot(self):
        print(self.omega)
        self.chad.plot()

    def graph(self):
        plt.rcParams["figure.figsize"] = (50, 40)

        nodes, edges, labels = gp.graph(self.omega)
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        pos = graphviz_layout(g)

        nx.draw_networkx_nodes(g, pos, node_size=20000, node_color='grey')
        nx.draw_networkx_edges(g, pos)
        nx.draw_networkx_labels(g, pos, labels, font_size=50)
        plt.savefig('real_test_run_1.png')
        plt.show()
