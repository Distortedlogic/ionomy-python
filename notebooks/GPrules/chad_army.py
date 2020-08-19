from os import stat
import pickle, time, random

from deap import tools, gp

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

class ChadArmy:
    def __init__(self, chad, pset, toolbox, stats):
        self.chad = chad
        self.pset = pset
        self.toolbox = toolbox
        self.stats = stats
        self.round = 0

    def load_pop(self, pop_size):
        self.gen = 0
        cp = self.load(f"init_pop_{pop_size}.pkl")
        pop = cp["pop"]
        if len(pop) != pop_size:
            raise Exception
        self.halloffame = cp["halloffame"]
        self.logbook = cp["logbook"]
        random.setstate(cp["rndstate"])
        self.full_stats()
        return pop

    def filter_positives(self, pop):
        filtered = []
        selected = self.toolbox.fitness_gt(pop)
        while len(filtered) < len(pop):
            filtered.append(self.toolbox.clone(selected))
        return pop[:len(pop)]

    def init_pop(self, pop_size):
        random.seed(7)
        self.logbook = tools.Logbook()
        self.logbook.header = ['gen', 'round', 'nevals', 'runtime', 'cast'] + (self.stats.fields)
        self.halloffame = tools.HallOfFame(1)
        pop = self.toolbox.population(n=pop_size)
        self.evaluate_pop(pop, "Init")
        cp = dict(
            pop=pop,
            halloffame=self.halloffame,
            logbook=self.logbook,
            rndstate=random.getstate()
        )
        self.checkpoint(cp, f"init_pop_{pop_size}.pkl")
        return pop

    def war(self, ngen, pop_size):
        try:
            pop = self.load_pop(pop_size)
        except Exception as e:
            print(e)
            pop = self.init_pop(pop_size)
        for self.gen in range(1, ngen + 1):
            self.round = 1
            pop = self.standard_step(pop, 0.5, 0.7)
            if self.gen > 10 and self.gen % 5 == 0:
                pop = self.toolbox.selDoubleTournament(pop, len(pop), fitness_size=len(pop)//5)
            else:
                pop = self.toolbox.selTournament(pop, len(pop), tournsize=len(pop)//5)
                # pop = self.filter_positives(pop)
            self.checkpoint(
                dict(
                    pop=pop,
                    halloffame=self.halloffame,
                    logbook=self.logbook,
                    gen=self.gen,
                    rndstate=random.getstate()
                ), f"checkpoints/pop_{self.gen}.pkl"
            )
        self.omega = self.halloffame[0]

    def standard_step(self, pop, cxpb, mutpb):
        offspring = [self.toolbox.clone(ind) for ind in pop]
        for i in range(1, len(offspring), 2):
            if random.random() < cxpb:
                offspring[i - 1], offspring[i] = self.toolbox.mate(
                    offspring[i - 1],
                    offspring[i]
                )
                del offspring[i - 1].fitness.values, offspring[i].fitness.values

        for i in range(len(offspring)):
            if random.random() < mutpb:
                if random.random() < 0.3:
                    offspring[i], = self.toolbox.mutEphemeral_rand(offspring[i])
                else:
                    offspring[i], = self.toolbox.mutUniform(offspring[i])
                del offspring[i].fitness.values
        self.evaluate_pop(offspring, "S")
        return offspring

    def custom_step(self, pop):
        # pop = self.apply_func(pop, self.node_mut, "Node", 10)
        pop = self.apply_func(pop, self.ephem_mut, "Ephem", 10)
        pop = self.apply_func(pop, self.shrink_mut, "Shrink", 5)
        pop = self.apply_func(pop, self.mate, "Mate", 5)
        best = [self.toolbox.clone(ind) for ind in self.toolbox.selBest(pop, len(pop) // 10)]
        pop = self.toolbox.selTournament(pop, len(pop) - len(pop) // 10, tournsize=2) + best
        pop = self.keep_best_apply(pop, len(pop) // 10, self.uniform_mut, "Uniform")
        return pop

    def log(self, pop, nevals, cast):
        record = self.stats.compile(pop)
        mins, secs = divmod(int(time.time() - self.start_time), 60)
        self.logbook.record(
            gen=self.gen,
            round=self.round,
            nevals=nevals,
            runtime=f'{mins} mins {secs} secs',
            cast=cast,
            **record)
        print(self.logbook.stream)

    def evaluate_pop(self, pop, cast):
        self.start_time = time.time()
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        self.log(pop, len(invalid_ind), cast)
        try:
            prev = self.halloffame[0]
        except:
            self.halloffame.update(pop)
            self.full_stats()
        else:
            self.halloffame.update(pop)
            if prev != self.halloffame[0]:
                self.full_stats()

    def apply_func(self, pop, func, cast, epochs):
        for _ in range(epochs):
            offspring = [self.toolbox.clone(ind) for ind in pop]
            offspring = func(offspring)
            self.evaluate_pop(offspring, cast)
            pop = self.best_of_two(offspring, pop)
            self.log(pop, 0, "best_of_two")
            self.round += 1
        return pop

    @staticmethod
    def better(competitors):
        ind1, ind2 = competitors
        if ind1.fitness.values[0] >= ind2.fitness.values[0]:
            return ind1
        else:
            return ind2

    def best_of_two(self, pop1, pop2):
        return self.toolbox.map(self.better, zip(pop1, pop2))

    def keep_best_apply(self, pop, keep_best, func, cast):
        best = [self.toolbox.clone(ind) for ind in self.toolbox.selBest(pop, keep_best)]
        offspring = func(pop)
        self.evaluate_pop(offspring, cast)
        return self.toolbox.selBest(offspring, len(offspring) - keep_best) + best

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

    def full_stats(self):
        self.chad.fitness(self.halloffame[0], full_stats=True)
        self.chad.plot_ec()
        self.chad.profit_bar()
        self.chad.print_results()
        self.chad.plot_trades()
        self.graph(self.halloffame[0])
        self.chad.print_history()
    def load(self, path):
        with open(path, "rb") as cp_file:
            cp = pickle.load(cp_file)
        return cp

    def checkpoint(self, cp, path):
        with open(path, "wb") as cp_file:
            pickle.dump(cp, cp_file)

    def insert_mut(self, pop):
        for i in range(len(pop)):
            pop[i], = self.toolbox.mutInsert(pop[i], pset = self.pset)
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