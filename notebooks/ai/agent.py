from statistics import mean

import numpy as np
import pandas as pd
import pandas_ta as ta
from pandas.core.frame import DataFrame
from pandas.core.series import Series

from .strategy import Deep_Evolution_Strategy
from .model import Model
from .environment import Environment

FEE_RATE = 0.003

class Agent:
    def __init__(
        self,
        ohlcv_df: DataFrame,
        window_size: int,
        num_layers: int,
        layer_size: int,
        output_size: int,
        population_size: int,
        sigma: float,
        learning_rate: float,
        initial_capital: float,
        max_buy: int,
        max_sell: int
    ):
        self.window_size = window_size
        self.initial_capital = initial_capital
        self.max_buy = max_buy
        self.max_sell = max_sell

        self.ohlcv_df = ohlcv_df
        self.env = Environment(ohlcv_df, [{"name": "ema", "length":9}])
        self.model = Model(self.env.dim * window_size, num_layers, layer_size, output_size)
        self.stategy = Deep_Evolution_Strategy(
            self.model.weights,
            self.get_reward,
            population_size,
            sigma,
            learning_rate,
        )

    def act(self, state: np.ndarray):
        decision, amount = self.model.predict(state)
        try:
            decision, amount = np.argmax(decision[0]), int(amount[0])
        except:
            return 0, 0
        return decision, amount

    def get_reward(self, weights: np.ndarray) -> float:
        self.model.weights = weights
        balance = self.initial_capital
        position = 0

        for time_index in range(0, self.env.length):
            state = self.env.get_state(time_index, self.window_size)
            signal, amount = self.act(state)
            amount = int(np.around(amount))
            if amount < 1:
                continue
            current_price = self.env.close[time_index]
            if signal == 1:
                buy_units = min(amount, self.max_buy)
                total_buy = buy_units * current_price
                if total_buy > balance:
                    continue
                fee = FEE_RATE*total_buy
                balance -= (total_buy + fee)
                position += buy_units

            elif signal == 2 and position > 0:
                sell_units = min(amount, position, self.max_sell)
                position -= sell_units
                total_sell = sell_units * current_price
                fee = FEE_RATE*total_sell
                balance += (total_sell - fee)

        return ((balance - self.initial_capital) / self.initial_capital) * 100

    def simulate(self):
        balance = self.initial_capital
        buy_orders = pd.DataFrame(columns=["amount", "price"])
        buy_history = pd.DataFrame(columns=[
            "time_index", "amount", "price", "balance", "position", "fee"
        ])
        sell_history = pd.DataFrame(columns=[
            "time_index", "amount", "price", "balance", "position", "fee", "roi", "entry_price"
        ])
        position = 0
        actions = []
        for time_index in range(0, self.env.length):
            if time_index < self.window_size:
                continue
            state = self.env.get_state(time_index, self.window_size + 1)
            signal, amount = self.act(state)
            amount = int(np.around(amount))
            if amount < 1:
                continue
            actions.append([signal, amount])
            current_price = self.env.close[time_index]
            if signal == 1:
                buy_units = min(amount, self.max_buy)
                total_buy = buy_units * current_price
                if total_buy > balance:
                    continue
                fee = FEE_RATE*total_buy
                balance -= (total_buy + fee)
                position += buy_units

                buy_order = {"amount": buy_units, "price": current_price}
                buy_orders = buy_orders.append(buy_order, ignore_index=True)
                row = {
                    "time_index": time_index,
                    "amount": buy_units,
                    "price": current_price,
                    "balance": balance,
                    "position": position,
                    "fee": fee
                }
                buy_history = buy_history.append(row, ignore_index=True)
            elif signal == 2 and position > 0:
                sell_units = min(amount, position, self.max_sell)
                position -= sell_units
                total_sell = sell_units * current_price
                fee = FEE_RATE*total_sell
                balance += (total_sell - fee)

                buy_orders.sort_values(by="price", ascending=True, inplace=True)
                order_amount = sell_units
                entry_orders = pd.DataFrame(columns=["amount", "price"])
                for index, row in buy_orders.iterrows():
                    if row["amount"] < order_amount:
                        buy_orders.drop(index=index)
                        order_amount -= row["amount"]
                        order = {"amount": row["amount"], "price": row["price"]}
                        entry_orders = entry_orders.append(order, ignore_index=True)
                    else:
                        buy_orders.loc[index, "amount"] -= order_amount
                        order = {"amount": order_amount, "price": row["price"]}
                        entry_orders = entry_orders.append(order, ignore_index=True)
                        break
                    

                entry_capital = entry_orders["amount"].multiply(entry_orders["price"]).sum()
                entry_price = entry_capital/entry_orders["amount"].sum()
                try:
                    roi = ((total_sell - entry_capital) / entry_capital)* 100
                except:
                    roi = 0

                row = {
                    "time_index": time_index,
                    "amount": sell_units,
                    "price": current_price,
                    "balance": balance,
                    "position": position,
                    "entry_price": entry_price,
                    "fee": fee,
                    "roi": roi
                }
                sell_history = sell_history.append(row, ignore_index=True)
        return buy_history, sell_history
