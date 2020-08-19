import json
import time

import IPython
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from deap import gp
from networkx.drawing.nx_agraph import graphviz_layout
from scipy.stats import kstest

hof_log_cols = [
    "Size",
    "Execution Time",
    "avg_ROI",
    "Total Profit",
    "Max Profit",
    "Min Profit",
    "Mean Profit",
    "Root Number Trades",
    "Standard Deviation",
    "System Quality Number",
    "Number Trades",
    "Number Longs",
    "Number Shorts",
    "Number Positive",
    "Sum Positive",
    "Number Negative",
    "Sum Negative",
    "Number Not Closed",
    "closed_ratio",
    "mean_entry_time_diff",
    "trade_duration",
    "trade_time_coeff"
]

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

class Chad:
    def __init__(self, pset, df):
        self.pset = pset
        self.df = df
        self.hof_log = pd.DataFrame(columns=hof_log_cols)
    def fitness(self, individual, full_stats=False):
        start_time = time.time()
        history = pd.DataFrame(columns=cols)

        # try:
        signals, min_hold = gp.compile(individual, self.pset)(self.df)
        self.prices = signals['close']

        position = 0
        max_buy = 0.1
        max_sell = 0.1
        FEE_RATE = 0.002
        hold_time_index = -1

        for time_index, row in signals.iterrows():
            # if time_index < hold_time_index:
            #     continue
            current_price = row['close']
            if row['buy'] and position >= -1 and position < 1:
                total_buy = max_buy * current_price
                position += 1
                buy_fee = FEE_RATE * total_buy
                if history.empty or history.loc[history['type'] == 'short', 'closed'].all():
                    record = {
                        "entry_time": time_index,
                        "entry": total_buy,
                        "entry_fee": buy_fee,
                        "type": "long",
                        "closed": False,
                        "exit": None,
                        "exit_time": None,
                        "profit": None
                    }
                    history = history.append(record, ignore_index=True)
                    hold_time_index = time_index + min_hold
                else:
                    mask = (history['type'] == 'short') & (history['closed'] == False)
                    idx = history[mask].index[0]
                    history.loc[idx, 'closed'] = True
                    history.loc[idx, 'exit'] = total_buy
                    history.loc[idx, 'exit_time'] = time_index
                    history.loc[idx, 'exit_fee'] = buy_fee
                    trade = history.iloc[idx]
                    total_fee = trade['entry_fee'] + trade['exit_fee']
                    gross_profit = trade['entry'] - total_buy
                    history.loc[idx, 'profit'] = gross_profit - total_fee
            if row['sell'] and position > -1 and position <= 1:
                total_sell = max_sell * current_price
                position -= 1
                if history.empty or history.loc[history['type'] == 'long', 'closed'].all():
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
                    hold_time_index = time_index + min_hold
                else:
                    mask = (history['type'] == 'long') & (history['closed'] == False)
                    idx = history[mask].index[0]
                    history.loc[idx, 'closed'] = True
                    history.loc[idx, 'exit'] = total_sell
                    history.loc[idx, 'exit_time'] = time_index
                    history.loc[idx, 'exit_fee'] = FEE_RATE * total_sell
                    trade = history.iloc[idx]
                    total_fee = trade['exit_fee'] + trade['entry_fee']
                    gross_profit = total_sell - trade['entry']
                    history.loc[idx, 'profit'] = gross_profit - total_fee
        self.history = history
        history = history[history['closed'] == True]
        num_trades = len(history)
        if num_trades < 30 or history.empty or history['profit'].std() == 0:
            return -99999 + num_trades,
        mean_entry_time_diff = history['entry_time'].sub(history['entry_time'].shift(1)).dropna().mean()
        trade_duration = history['exit_time'].sub(history['entry_time']).mean()
        trade_time_coeff = mean_entry_time_diff / trade_duration
        # times = pd.concat([history['entry_time'], history['exit_time'], pd.Series([0]), pd.Series([len(self.prices)])])
        # statistic, _ = kstest(times.to_list(), "uniform")
        # time_const = root_avg_trade_duration * statistic
        num_not_closed = len(self.history[self.history['closed'] == False])
        closed_ratio = (num_trades / (num_not_closed + num_trades))
        self.mean_profit = history['profit'].mean()
        self.std_profit = history['profit'].std()
        self.root_num_trades = num_trades ** 0.7
        amplifier = self.root_num_trades / self.std_profit
        if self.mean_profit < 0:
            amplifier = 1/amplifier
        self.SQN = amplifier * self.mean_profit
        if full_stats:
            self.history['hold_time'] = history['exit_time'].sub(history['entry_time'])
            self.avg_hold_time = self.history['hold_time'].mean()
            self.avg_ROI = history['profit'].divide(history['entry']).mean()
            self.amplifier = amplifier
            self.num_trades = num_trades
            self.total_profit = history['profit'].sum()
            self.max_profit = history["profit"].max()
            self.min_profit = history["profit"].min()
            self.num_longs = len(history[history['type'] == "long"])
            self.num_shorts = len(history[history['type'] == "short"])
            self.sum_pos = history.loc[history['profit'] > 0, "profit"].sum()
            self.num_pos = len(history[history['profit'] > 0])
            self.sum_neg = history.loc[history['profit'] < 0, "profit"].sum()
            self.num_neg = len(history[history['profit'] < 0])
            self.num_not_closed = num_not_closed
            self.size = len(individual)
            self.closed_ratio = closed_ratio
            self.mean_entry_time_diff = mean_entry_time_diff
            self.trade_duration = trade_duration
            self.trade_time_coeff = trade_time_coeff
            self.exec_time = int(time.time() - start_time)
        return self.SQN,
    
    @staticmethod
    def graph(ind):
        plt.rcParams["figure.figsize"] = (150, 40)

        nodes, edges, labels = gp.graph(ind)
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        pos = graphviz_layout(g)

        nx.draw_networkx_nodes(g, pos, node_size=20000, node_color='grey')
        nx.draw_networkx_edges(g, pos)
        nx.draw_networkx_labels(g, pos, labels, font_size=50)
        plt.savefig('graph.png')
        plt.show()

    def plot_trades(self):
        fig = plt.figure(figsize = (15,5))
        prices = self.df['close'][250:].reset_index(drop=True)
        history = self.history[self.history['closed'] == True]
        plt.plot(prices, color='black', lw=2.)
        long_buys = history.loc[history['type'] == 'long', 'entry_time'].sort_values().astype(int).to_list()
        short_buys = history.loc[history['type'] == 'short', 'exit_time'].sort_values().astype(int).to_list()
        long_sells = history.loc[history['type'] == 'long', 'exit_time'].sort_values().astype(int).to_list()
        short_sells = history.loc[history['type'] == 'short', 'entry_time'].sort_values().astype(int).to_list()
        plt.plot(prices, '^', markersize=10, color='lime', label = 'long buy', markevery = long_buys)
        plt.plot(prices, '^', markersize=10, color='darkgreen', label = 'short buy', markevery = short_buys)
        plt.plot(prices, 'v', markersize=10, color='orangered', label = 'long sell', markevery = long_sells)
        plt.plot(prices, 'v', markersize=10, color='darkred', label = 'short sell', markevery = short_sells)
        plt.legend()
        plt.grid()
        # plt.savefig('trades.png')
        plt.show()

    def plot_ec(self):
        fig = plt.figure(figsize = (15,5))
        equity = self.history.loc[self.history['closed'], 'profit'].cumsum()
        plt.plot(equity)
        plt.savefig('equity_curve.png')
        plt.show()

    def profit_bar(self):
        profits = self.history.loc[self.history["closed"], "profit"]
        lower = int(self.mean_profit - 3 * self.std_profit)
        upper = int(self.mean_profit + 3 * self.std_profit)
        step = int(self.std_profit) // 3
        bins = sorted(list(range(lower, upper, step)) + [0])
        cuts = pd.cut(profits, bins=bins, include_lowest=True, duplicates='drop')
        ax = cuts.value_counts(sort=False).plot.bar(rot=0, color="b", figsize=(15,5))
        plt.xticks(rotation=90)
        plt.savefig('profit_bar.png')
        plt.show()

    def print_history(self):
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            IPython.display.display(self.history)

    def print_results(self):
        try:
            mins, secs = divmod(self.exec_time, 60)
        except:
            pass
        else:
            record = {
                "Size": self.size,
                "avg_ROI": self.avg_ROI,
                "Execution Time": f'{mins} mins {secs} secs',
                "Total Profit": self.total_profit,
                "Mean Profit": self.mean_profit,
                "Max Profit": self.max_profit,
                "Min Profit": self.min_profit,
                "Standard Deviation": self.std_profit,
                "Number Trades": self.num_trades,
                "Root Number Trades": self.root_num_trades,
                "Number Longs": self.num_longs,
                "Number Shorts": self.num_shorts,
                "Number Positive": self.num_pos,
                "Sum Positive": self.sum_pos,
                "Number Negative": self.num_neg,
                "Sum Negative": self.sum_neg,
                "Number Not Closed": self.num_not_closed,
                "System Quality Number": self.SQN,
                "closed_ratio": self.closed_ratio,
                "mean_entry_time_diff": self.mean_entry_time_diff,
                "trade_duration": self.trade_duration,
                "trade_time_coeff": self.trade_time_coeff,
                "Average Hold Time": self.avg_hold_time
            }
            self.hof_log = self.hof_log.append(record, ignore_index=True)
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                IPython.display.display(self.hof_log[-25:])
