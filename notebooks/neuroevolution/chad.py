import json
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .environment import Environment
from .model import Model

FEE_RATE = 0.003
HOLD_DISCOUNT = 0.05

class Chad:
    def __init__(
        self,
        network_size: int,
        output_size: int,
        env: Environment
    ) -> None:
        self.env = env
        self.network_size = network_size
        self.output_size = output_size
    
    def __len__(self):
        return len(Model(self.env.window_size, self.network_size, self.output_size).flatten())

    def fitness(self, individual):
        model = Model(self.env.window_size, self.network_size, self.output_size)
        model.set_weights(np.asarray(individual))
        balance = self.env.initial_capital
        position = 0

        order_cols = ['time', "price", "amount"]
        orders = pd.DataFrame(columns=order_cols)
        buy_cols = ["time_index", "price", "amount", "balance", "fee"]
        buy_history = pd.DataFrame(columns=buy_cols)
        sell_cols = ["time_index", "price", "amount", "balance", "fee", "position", "entry", "profit", "roi"]
        sell_history = pd.DataFrame(columns=sell_cols)

        alive = True
        lifespan = 0
        num_trades = 0
        for time_index in range(self.env.window_size, self.env.length):
            lifespan = time_index
            if balance < self.env.initial_capital/2 or position < 0:
                alive = False
                break
            state = self.env.get_state(time_index)
            signal = model.predict(state)
            current_price = self.env.close[time_index]
            record = {"time_index": time_index, "price": current_price}
            if signal == 1:
                buy_units = self.env.max_buy
                total_buy = buy_units * current_price
                if total_buy > balance:
                    continue
                fee = FEE_RATE*total_buy
                balance -= (total_buy + fee)
                position += buy_units
                record = {**record, "balance": balance, "fee": fee, "amount": buy_units}
                buy_history = buy_history.append(record, ignore_index=True)
                orders = orders.append(
                    {'time':time_index, "price": current_price, "amount": buy_units},
                    ignore_index=True
                )
                num_trades += 1
            elif signal == 2 and position > 0:
                sell_units = min(position, self.env.max_sell)
                record = {**record, "amount": sell_units}
                total_sell = sell_units * current_price
                fee = FEE_RATE*total_sell
                balance += (total_sell - fee)
                entry_cols = ["price", "amount"]
                entry_orders = pd.DataFrame(columns=entry_cols)
                for index, row in orders.iterrows():
                    if row["amount"] > sell_units:
                        position -= sell_units
                        entry = {"price": row["price"], "amount": sell_units}
                        entry_orders = entry_orders.append(entry, ignore_index=True)
                        orders.loc[index, "amount"] -= sell_units
                        break
                    elif row["amount"] == sell_units:
                        position -= sell_units
                        entry = {"price": row["price"], "amount": sell_units}
                        entry_orders = entry_orders.append(entry, ignore_index=True)
                        orders.drop(index=index)
                        break
                    else:
                        position -= row["amount"]
                        entry = {"price": row["price"], "amount": row["amount"]}
                        entry_orders = entry_orders.append(entry, ignore_index=True)
                        orders.drop(index=index)
                record = {**record, "balance": balance, "position": position, "fee": fee}
                entry_sum = entry_orders["price"].multiply(entry_orders["amount"]).sum()
                entry_price = entry_sum/entry_orders["amount"].sum()
                profit = total_sell - entry_sum
                roi = profit/entry_sum
                record = {**record, "entry": entry_price, "profit": profit, "roi": roi}
                sell_history = sell_history.append(record, ignore_index=True)
                num_trades += 1
            if not orders.empty:
                for index, order in orders.iterrows():
                    time, price, amount = itemgetter('time', 'price', 'amount')(order)
                    if time_index - time > 24:
                        position -= amount
                        orders.drop(index)
        if num_trades < 6:
            alive = False
            lifespan = 0
        position_value = (1-HOLD_DISCOUNT) * position * self.env.close.iloc[-1]
        total = balance + position_value
        roi = (total - self.env.initial_capital)/self.env.initial_capital
        results = {
            "balance": balance,
            "position": position,
            "last": self.env.close.iloc[-1],
            "position_value": position_value,
            "total": total,
            "roi": roi,
            
        }
        self.buy_history = buy_history
        self.sell_history = sell_history
        self.results = results
        return roi if alive else -1, lifespan/self.env.length

    def plot(self, buy_history, sell_history, results):
        print(results)
        fig = plt.figure(figsize = (15,5))
        plt.plot(self.env.close, color='r', lw=2.)
        buys = buy_history["time_index"].astype(int).to_list()
        sells = sell_history["time_index"].astype(int).to_list()
        plt.plot(self.env.close, '^', markersize=10, color='m', label = 'buying signal', markevery = buys)
        plt.plot(self.env.close, 'v', markersize=10, color='k', label = 'selling signal', markevery = sells)
        plt.legend()
        plt.show()
