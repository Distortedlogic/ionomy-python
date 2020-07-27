import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import gp

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from scipy.stats import kstest


class Chad:
    def __init__(self, pset, df):
        self.pset = pset
        self.df = df
    def fitness(self, individual):
        cols = [
            "entry",
            "exit",
            "entry_fee",
            "type",
            "closed",
            "entry_time",
            "exit_time",
            "profit"
        ]
        history = pd.DataFrame(columns=cols)

        # try:
        signals = gp.compile(individual, self.pset)(self.df)
        # except:
        #     print(individual)
        #     self.graph(ind=individual)

        data = pd.concat([signals, self.df['close']], axis=1)
        data = data[250:].reset_index(drop=True)
        self.prices = data['close']

        position = 0
        max_buy = 0.1
        max_sell = 0.1
        FEE_RATE = 0.002

        for time_index, row in data.iterrows():
            current_price = row['close']
            if row['buy']:
                total_buy = max_buy * current_price
                position += max_buy
                if history.empty or history['closed'].all():
                    record = {
                        "entry_time": time_index,
                        "entry": total_buy,
                        "entry_fee": FEE_RATE * total_buy,
                        "type": "long",
                        "closed": False,
                        "exit": None,
                        "exit_time": None,
                        "profit": None
                    }
                    history = history.append(record, ignore_index=True)
                else:
                    idx = history[history['closed'] == False].index[0]
                    history.loc[idx, 'closed'] = True
                    history.loc[idx, 'exit'] = total_buy
                    history.loc[idx, 'exit_time'] = time_index
                    history.loc[idx, 'exit_fee'] = FEE_RATE * total_buy
                    trade = history.loc[idx]
                    history.loc[idx, 'profit'] = trade['entry'] - total_buy
            if row['sell']:
                total_sell = max_sell * current_price
                position -= max_sell
                if history.empty or history['closed'].all():
                    record = {
                        "entry_time": time_index,
                        "entry": total_sell,
                        "entry_fee": FEE_RATE * total_sell,
                        "type": "short",
                        "closed": False,
                        "exit": None,
                        "exit_time": None,
                        "profit": None
                    }
                    history = history.append(record, ignore_index=True)
                else:
                    idx = history[history['closed'] == False].index[0]
                    history.loc[idx, 'closed'] = True
                    history.loc[idx, 'exit'] = total_sell
                    history.loc[idx, 'exit_time'] = time_index
                    history.loc[idx, 'exit_fee'] = FEE_RATE * total_sell
                    trade = history.loc[idx]
                    history.loc[idx, 'profit'] = total_sell - trade['entry']
        self.history = history
        history = history[history['closed']]
        num_trades = len(history)
        if num_trades < 30 or history.empty or history['profit'].std() == 0:
            return -99999 + num_trades,
        root_avg_trade_duration = history['exit_time'].sub(history['entry_time']).mean() ** 0.5
        times = pd.concat([history['entry_time'], history['exit_time'], pd.Series([0]), pd.Series([len(self.prices)])])
        statistic, _ = kstest(times.to_list(), "uniform")
        time_const = root_avg_trade_duration * statistic
        mean_profit = history['profit'].mean()
        amplifier = time_const * (num_trades ** 0.5) / history['profit'].std()
        if mean_profit < 0:
            amplifier = 1/amplifier
        SQN = amplifier * mean_profit
        return SQN,
    
    @staticmethod
    def graph(ind):
        plt.rcParams["figure.figsize"] = (50, 40)

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

    def plot(self):
        fig = plt.figure(figsize = (15,5))
        prices = self.prices
        plt.plot(prices, color='r', lw=2.)
        buys = pd.concat([
            self.history.loc[(self.history['type'] == 'long') & ~(self.history['entry_time'].isnull()), 'entry_time'],
            self.history.loc[(self.history['type'] == 'short') & ~(self.history['exit_time'].isnull()), 'exit_time']
        ]).sort_values().astype(int).to_list()
        sells = pd.concat([
            self.history.loc[(self.history['type'] == 'long') & ~(self.history['exit_time'].isnull()), 'exit_time'],
            self.history.loc[(self.history['type'] == 'short') & ~(self.history['entry_time'].isnull()), 'entry_time']
        ]).sort_values().astype(int).to_list()
        plt.plot(prices, '^', markersize=10, color='m', label = 'buying signal', markevery = buys)
        plt.plot(prices, 'v', markersize=10, color='k', label = 'selling signal', markevery = sells)
        plt.legend()
        plt.show()