import json
from math import exp
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.special import expit

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
        self.one_per_hour_opt = self.env.length/12
        self.max_x = -scipy.optimize.minimize(
            self.num_trades_coeff,
            0
        ).fun

    def num_trades_coeff(self, x):
        return -1 * exp(x-x**2/self.one_per_hour_opt)
    
    def __len__(self):
        return len(Model(self.env.window_size * self.env.num_features, self.network_size, self.output_size).flatten())

    def fitness(self, individual):
        model = Model(self.env.window_size * self.env.num_features, self.network_size, self.output_size)
        model.set_weights(np.asarray(individual))
        balance = self.env.initial_capital
        position = 0

        order_cols = ['time', "price", "amount"]
        orders = pd.DataFrame(columns=order_cols)
        buy_cols = ["time_index", "price", "amount", "balance", "fee"]
        buy_history = pd.DataFrame(columns=buy_cols)
        sell_cols = ["time_index", "price", "amount", "balance", "fee", "position", "entry", "profit", "roi"]
        sell_history = pd.DataFrame(columns=sell_cols)

        total_profit = 0
        num_trades = 0
        for time_index in range(self.env.window_size, self.env.length):
            state = self.env.get_state(time_index)
            try:
                signal = model.predict(state, position)
            except Exception as e:
                print(self.env.df.loc[time_index+2-self.env.window_size:time_index+1])
                raise e
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
            elif signal == 2:
                if position <= 0.09:
                    continue
                sell_units = min(position, self.env.max_sell)
                record = {**record, "amount": sell_units}
                total_sell = sell_units * current_price
                fee = FEE_RATE * total_sell
                balance += (total_sell - fee)
                position -= sell_units
                record = {**record, "fee": fee, "balance": balance, "position": position}
                order = orders.iloc[0]
                entry_sum = order['price'] * order['amount']
                orders = orders.iloc[1:]
                time_discounted = 1 - expit(order['time'] - time_index - 24)
                profit = (1 - 2 * FEE_RATE) * time_discounted * (total_sell - entry_sum)
                total_profit += profit
                roi = profit/entry_sum
                record = {**record, "entry": order['price'], "profit": profit, "roi": roi}
                sell_history = sell_history.append(record, ignore_index=True)
                num_trades += 1
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
        return total_profit * -self.num_trades_coeff(num_trades)/self.max_x,

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


                # entry_cols = ["price", "amount"]
                # entry_orders = pd.DataFrame(columns=entry_cols)
                # for index, row in orders.iterrows():
                #     if row["amount"] > sell_units:
                #         position -= sell_units
                #         entry = {"price": row["price"], "amount": sell_units}
                #         entry_orders = entry_orders.append(entry, ignore_index=True)
                #         orders.loc[index, "amount"] -= sell_units
                #         break
                #     elif row["amount"] == sell_units:
                #         position -= sell_units
                #         entry = {"price": row["price"], "amount": sell_units}
                #         entry_orders = entry_orders.append(entry, ignore_index=True)
                #         orders.drop(index=index)
                #         break
                #     else:
                #         position -= row["amount"]
                #         entry = {"price": row["price"], "amount": row["amount"]}
                #         entry_orders = entry_orders.append(entry, ignore_index=True)
                #         orders.drop(index=index)
                # record = {**record, "balance": balance, "position": position, "fee": fee}
                # entry_sum = entry_orders["price"].multiply(entry_orders["amount"]).sum()
                # entry_price = entry_sum/entry_orders["amount"].sum()
