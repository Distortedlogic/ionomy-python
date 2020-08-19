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
from .utils.mutNodeReplacement import mutNodeReplacement
from .utils.mutInsert import mutInsert

def mutEphemeral_rand(individual):
    ephemerals_idx = [
        index for index, node in enumerate(individual) if isinstance(node, Ephemeral)
    ]
    if len(ephemerals_idx) > 0:
        ephemerals_idx = random.sample(ephemerals_idx, random.randint(1, len(ephemerals_idx)))
        for i in ephemerals_idx:
            individual[i] = type(individual[i])()

    return individual,

def fitness_gt(pop, limit):
    chosen = []
    for i in range(len(pop)):
        if pop[i].fitness.values[0] > limit:
            chosen.append(pop[i])
    return chosen

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
            parsimony_size=1.4,
            fitness_first=True
        )
        toolbox.register("selBest", tools.selBest)
        toolbox.register("fitness_gt", fitness_gt)
        toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
        toolbox.register("mutUniform", gp.mutUniform, expr=toolbox.expr, pset=pset)
        toolbox.register("mutNodeReplacement", mutNodeReplacement, pset=pset)
        toolbox.register("mutEphemeral", gp.mutEphemeral, mode='all')
        toolbox.register("mutEphemeral_rand", mutEphemeral_rand)
        toolbox.register("mutInsert", mutInsert)
        toolbox.register("mutShrink", gp.mutShrink)

        max_height = 30
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_height))
        toolbox.decorate("mutUniform", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_height))
        toolbox.decorate("mutNodeReplacement", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_height))
        toolbox.decorate("mutEphemeral", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_height))
        toolbox.decorate("mutEphemeral_rand", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_height))

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

    def graph(self):
        self.army.graph()
