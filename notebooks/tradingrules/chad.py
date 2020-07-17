import json
import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt

class Chad:
    def __init__(self, env):
        self.env = env
    def fitness(self, individual):
        order_cols = ['time', "price", "amount"]
        orders = pd.DataFrame(columns=order_cols)
        buy_cols = ["time_index", "price", "amount", "balance", "fee"]
        buy_history = pd.DataFrame(columns=buy_cols)
        sell_cols = ["time_index", "price", "amount", "balance", "fee", "position", "entry", "profit", "roi"]
        sell_history = pd.DataFrame(columns=sell_cols)

        self.signals = self.env.explore(individual)

        position = 0
        initial_capital = 10_000
        balance = 10_000
        total_profit = 0
        num_trades = 0
        max_buy = 0.1
        max_sell = 0.1
        FEE_RATE = 0.002

        for time_index, row in self.signals.iterrows():
            current_price = row['close']
            record = {"time_index": time_index, "price": current_price}
            if row['signal'] == 1 and position <= 0.3:
                buy_units = max_buy
                total_buy = buy_units * current_price
                fee = FEE_RATE * total_buy
                balance -= (total_buy + fee)
                position += buy_units
                record = {**record, "balance": balance, "fee": fee, "amount": buy_units}
                buy_history = buy_history.append(record, ignore_index=True)
                orders = orders.append(
                    {'time': time_index, "price": current_price, "amount": buy_units},
                    ignore_index=True
                )
            elif row['signal'] == -1 and position >= 0.09:
                sell_units = max_sell
                record = {**record, "amount": sell_units}
                total_sell = sell_units * current_price
                fee = FEE_RATE * total_sell
                balance += (total_sell - fee)
                position -= sell_units
                record = {
                    **record,
                    "fee": np.round(fee, decimals=3),
                    "balance": np.round(balance, decimals=3),
                    "position": np.round(position, decimals=3)
                }
                order = orders.iloc[0]
                entry_sum = order['price'] * order['amount']
                orders = orders.iloc[1:]
                profit = (1 - 2 * FEE_RATE) * (total_sell - entry_sum)
                total_profit += profit
                record = {
                    **record,
                    "entry": np.round(order['price'], decimals=3),
                    "profit": np.round(profit, decimals=3),
                    'total_profit': np.round(total_profit, decimals=3),
                    "roi": np.round(profit/entry_sum, decimals=3)
                }
                sell_history = sell_history.append(record, ignore_index=True)
                num_trades += 1

        total = balance + self.env.last * position
        self.buy_history = buy_history
        self.sell_history = sell_history
        self.results = {
            "balance": np.round(balance, decimals=3),
            "position": np.round(position, decimals=3),
            "total": np.round(total, decimals=3),
            "total_profit": np.round(total_profit, decimals=3),
            "roi": np.round(total / initial_capital, decimals=3)
        }
        std = sell_history['profit'].std()
        if num_trades < 30 or sell_history.empty or std == 0:
            return -30 + num_trades,
        SQN = (num_trades ** 0.5) * sell_history['profit'].mean() / std
        return SQN,

    def plot(self):
        print(json.dumps(self.results, indent=4))
        fig = plt.figure(figsize = (15,5))
        prices = self.env.prices()
        plt.plot(prices, color='r', lw=2.)
        buys = self.buy_history["time_index"].astype(int).to_list()
        sells = self.sell_history["time_index"].astype(int).to_list()
        plt.plot(prices, '^', markersize=10, color='m', label = 'buying signal', markevery = buys)
        plt.plot(prices, 'v', markersize=10, color='k', label = 'selling signal', markevery = sells)
        plt.legend()
        plt.show()