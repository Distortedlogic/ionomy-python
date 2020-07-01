import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from math import sqrt
from random import choice
from scipy.special import expit

from .environment import Environment
from .brain import Brain

FEE_RATE = 0.003

class Chad:
    def __init__(
        self,
        output_size: int,
        env: Environment
    ) -> None:
        self.env = env
        self.output_size = output_size
        self.tf_opt = 24
        self.life_span = 4000

    def time_discount(self, t):
        try:
            coeff = sqrt(1 - t ** 2 / (2 * self.tf_opt) ** 2) + 0.1
            return coeff
        except:
            return 0.1 / t
    
    def __len__(self):
        return Brain((self.env.window_size, self.env.num_features, 1), self.output_size).size()

    def fitness(self, individual, tf):
        position = 0
        brain = Brain((self.env.window_size, self.env.num_features, 1), self.output_size)
        brain.set_weights(np.asarray(individual))
        balance = self.env.initial_capital

        order_cols = ['time', "price", "amount"]
        orders = pd.DataFrame(columns=order_cols)
        buy_cols = ["time_index", "price", "amount", "balance", "fee"]
        buy_history = pd.DataFrame(columns=buy_cols)
        sell_cols = ["time_index", "price", "amount", "balance", "fee", "position", "entry", "profit", "roi"]
        sell_history = pd.DataFrame(columns=sell_cols)

        total_profit = 0
        num_trades = 0
        nt_opt = self.life_span/24
        if tf == None:
            existence = range(self.env.window_size, self.env.length)
        if tf == 'jittered':
            jitter = choice(range(self.env.window_size, self.env.length - self.life_span))
            existence = range(jitter, jitter + self.life_span)
        else:
            existence = range(tf, tf + self.life_span)
        for time_index in existence:
            state = self.env.get_state(time_index)
            signal = brain.predict(state, position)
            current_price = self.env.close[time_index]
            record = {"time_index": time_index, "price": current_price}
            if signal == 1 and position <= 0.3:
                buy_units = self.env.max_buy
                total_buy = buy_units * current_price
                fee = FEE_RATE*total_buy
                balance -= (total_buy + fee)
                position += buy_units
                record = {**record, "balance": balance, "fee": fee, "amount": buy_units}
                buy_history = buy_history.append(record, ignore_index=True)
                orders = orders.append(
                    {'time': time_index, "price": current_price, "amount": buy_units},
                    ignore_index=True
                )
            elif signal == 2 and position >= 0.09:
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
                time_discounted = self.time_discount(time_index)
                profit = (1 - 2 * FEE_RATE) * time_discounted * (total_sell - entry_sum)
                total_profit += profit
                roi = profit/entry_sum
                record = {**record, "entry": order['price'], "profit": profit, "roi": roi}
                sell_history = sell_history.append(record, ignore_index=True)
                num_trades += 1
        position_value = position * self.env.close.iloc[-1]
        total = balance + position_value
        roi = (total - self.env.initial_capital)/self.env.initial_capital
        results = {
            "balance": balance,
            "position": position,
            "last": self.env.close.iloc[-1],
            "position_value": position_value,
            "total": total,
            "profit": balance - self.env.initial_capital,
            "roi": roi,
        }
        self.buy_history = buy_history
        self.sell_history = sell_history
        self.results = results
        self.brain = brain
        try:
            num_trades_coeff = sqrt(1 - (num_trades - nt_opt) ** 2 / (nt_opt) ** 2) + 0.1
        except:
            num_trades_coeff = 0.1 / num_trades
        return total_profit * num_trades_coeff,

    def plot(self):
        print(self.results)
        fig = plt.figure(figsize = (15,5))
        plt.plot(self.env.close, color='r', lw=2.)
        buys = self.buy_history["time_index"].astype(int).to_list()
        sells = self.sell_history["time_index"].astype(int).to_list()
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
