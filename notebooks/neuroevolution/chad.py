from statistics import mean

import numpy as np
import pandas as pd
import pandas_ta as ta
from pandas.core.frame import DataFrame

from .model import Model
from .environment import Environment

FEE_RATE = 0.003 

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
        for time_index in range(self.env.length):
            if time_index < self.env.window_size:
                continue
            state = self.env.get_state(time_index)
            signal = model.predict(state)
            current_price = self.env.close[time_index]
            if signal == 1:
                total_buy = self.env.max_buy * current_price
                if total_buy > balance:
                    continue
                fee = FEE_RATE*total_buy
                balance -= (total_buy + fee)
                position += self.env.max_buy
            elif signal == 2 and position > 0:
                sell_units = min(position, self.env.max_sell)
                position -= sell_units
                total_sell = sell_units * current_price
                fee = FEE_RATE*total_sell
                balance += (total_sell - fee)
        return ((balance - self.env.initial_capital) / self.env.initial_capital),