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
from .chad_army import ChadArmy
from .primitives import build_pset
from .utils.safe_gen import generate_safe

from inspect import isclass
def mutInsert(individual, pset):
    """Inserts a new branch at a random position in *individual*. The subtree
    at the chosen position is used as child node of the created subtree, in
    that way, it is really an insertion rather than a replacement. Note that
    the original subtree will become one of the children of the new primitive
    inserted, but not perforce the first (its position is randomly selected if
    the new primitive has more than one child).

    :param individual: The normal or typed tree to be mutated.
    :returns: A tuple of one tree.
    """
    index = random.randrange(len(individual))
    node = individual[index]
    slice_ = individual.searchSubtree(index)
    choice = random.choice

    # As we want to keep the current node as children of the new one,
    # it must accept the return value of the current node
    primitives = [p for p in pset.primitives[node.ret] if node.ret in p.args]

    if len(primitives) == 0:
        return individual,

    new_node = choice(primitives)
    new_subtree = [None] * len(new_node.args)
    position = choice([i for i, a in enumerate(new_node.args) if a == node.ret])

    print(new_node.args)
    for i, arg_type in enumerate(new_node.args):
        if i != position:
            print(arg_type)
            print(pset.terminals)
            term = choice(pset.terminals[arg_type])
            if isclass(term):
                term = term()
            new_subtree[i] = term

    new_subtree[position:position + 1] = individual[slice_]
    new_subtree.insert(0, new_node)
    individual[slice_] = new_subtree
    return individual,

def mutEphemeral_rand(individual):
    ephemerals_idx = [
        index for index, node in enumerate(individual) if isinstance(node, Ephemeral)
    ]
    if len(ephemerals_idx) > 0:
        ephemerals_idx = random.sample(ephemerals_idx, random.randint(1, len(ephemerals_idx)))
        for i in ephemerals_idx:
            individual[i] = type(individual[i])()

    return individual,

class DemiChad:
    def __init__(self, df: DataFrame):
        self.df = df
        pset, terminal_types = build_pset()

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

        toolbox = base.Toolbox()
        toolbox.register("map", multiprocessing.Pool().map)
        toolbox.register("imap", multiprocessing.Pool().imap)
        toolbox.register("expr", generate_safe, pset=pset, min_=5, max_=10, terminal_types=terminal_types)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        toolbox.register(
            "selTournament",
            tools.selTournament
        )
        toolbox.register(
            "selDoubleTournament",
            tools.selDoubleTournament,
            fitness_size=25,
            parsimony_size=1.4,
            fitness_first=True
        )
        toolbox.register("selBest", tools.selBest)
        toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
        toolbox.register("mutUniform", gp.mutUniform, expr=toolbox.expr, pset=pset)
        toolbox.register("mutNodeReplacement", gp.mutNodeReplacement, pset=pset)
        toolbox.register("mutEphemeral", gp.mutEphemeral)
        toolbox.register("mutEphemeral_rand", mutEphemeral_rand)
        toolbox.register("mutInsert", mutInsert)
        toolbox.register("mutShrink", gp.mutShrink)

        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=20))
        toolbox.decorate("mutUniform", gp.staticLimit(key=operator.attrgetter("height"), max_value=20))
        toolbox.decorate("mutNodeReplacement", gp.staticLimit(key=operator.attrgetter("height"), max_value=20))
        toolbox.decorate("mutEphemeral", gp.staticLimit(key=operator.attrgetter("height"), max_value=20))
        toolbox.decorate("mutEphemeral_rand", gp.staticLimit(key=operator.attrgetter("height"), max_value=20))

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        self.stats = stats

        self.chad = Chad(pset, df)
        toolbox.register("evaluate", self.chad.fitness)
        self.pset = pset
        self.toolbox = toolbox

    def evolve(self, pop_size, ngen):
        self.army = ChadArmy(self.chad, self.pset, self.toolbox, self.stats)
        self.army.war(ngen, pop_size)
        self.chad.fitness(self.army.omega)

    def plot(self):
        print(self.army.omega)
        self.chad.plot()

    def graph(self):
        self.army.graph()
