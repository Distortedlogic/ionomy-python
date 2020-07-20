import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools

from .environment import Environment
from .chad import Chad
from .chad_army import ChadArmy

defaults = {
    'population_size': 500,
    'tournsize': 50,
    'mu': 0,
    'sigma': 0.025,
    'indpb': 0.05,
    'cxpb': 0.6,
    'mutpb': 0.6
}

class DemiChad:
    def __init__(self, ohlcv_df):
        self.ohlcv_df = ohlcv_df
        creator.create("FitnessMulti", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        self.create_tools()
    def create_tools(self):
        toolbox = base.Toolbox()
        toolbox.register("map", multiprocessing.Pool().map)

        fitness_stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        fitness_stats.register("f_avg", np.mean)
        fitness_stats.register("f_max", np.max)

        self.toolbox = toolbox
        self.fitness_stats = fitness_stats

    def evolve(self, ngen):
        self.env = Environment(self.ohlcv_df)
        self.army = ChadArmy(
            **defaults,
            env = self.env,
            toolbox = self.toolbox,
            fitness_stats = self.fitness_stats
        )
        best_SQN = self.army.war(ngen)
        self.omega()
        return best_SQN
    def omega(self):
        self.omega_chad = Chad(env = self.env)
        self.omega_chad.fitness(self.army.omega)
    def plot(self):
        self.env.print(self.army.omega)
        self.omega_chad.plot()

        # graph = networkx.DiGraph(self.army.history.genealogy_tree)
        # graph = graph.reverse()
        # colors = self.army.toolbox.map(
        #     self.army.toolbox.evaluate,
        #     (self.army.history.genealogy_history[i] for i in graph)
        # )
        # networkx.draw(graph, node_color=[color[0] for color in colors])
        # plt.show()