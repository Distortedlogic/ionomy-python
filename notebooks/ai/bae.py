import json
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from pandas.core.frame import DataFrame
from sqlalchemy import and_, create_engine, func
from sqlalchemy.orm.session import sessionmaker

from .agent import Agent
from .schema import AgentParams, Base

engine = create_engine('sqlite:///database.sqlite')
Session = sessionmaker(bind=engine)
Base.metadata.create_all(engine)

class Bae:
    def __init__(
        self,
        df: DataFrame,
        max_buy: int,
        max_sell: int,
        initial_capital: int,
        market: str
    ) -> None:
        self.df = df
        self.market = market
        self.db = Session()
        self.bayes_params = None
        self.config = {
            "initial_capital": initial_capital, 
            "max_buy": max_buy, 
            "max_sell": max_sell,
            "output_size": 3
        }
        
    def simulate_rewards(
        self,
        window_size: float,
        population_size: int,
        sigma: float,
        learning_rate: float,
        num_layers: int,
        layer_size: int,
        iterations: int = 100
    ):
        params = {
            "window_size": int(np.around(window_size)),
            "num_layers": int(np.around(num_layers)),
            "layer_size": int(np.around(layer_size)),
            "population_size": int(np.around(population_size)),
            "sigma": sigma,
            "learning_rate": learning_rate
        }
        print(params)
        agent = Agent(self.df, **params, **self.config)
        agent.stategy.train(iterations)
        return agent.stategy.reward_function(agent.stategy.weights, agent.stategy.bias)

    def optimize(
        self,
        init_points: int = 30,
        n_iter: int = 50,
        acq: str = 'ei',
        xi: float = 0.0
    ) -> None:
        NN_BAYESIAN = BayesianOptimization(
            self.simulate_rewards,
            {
                'window_size': (2, 50),
                'population_size': (1, 50),
                'sigma': (0.01, 0.99),
                'learning_rate': (0.000001, 0.49),
                'num_layers': (2, 10),
                'layer_size': (2, 100)
            },
            verbose=2
        )
        NN_BAYESIAN.maximize(init_points = init_points, n_iter = n_iter, acq = acq, xi = xi)
        params_df = pd.DataFrame.from_records([
            {"target": item["target"], **item["params"]} for item in NN_BAYESIAN.res
        ])
        params = params_df.iloc[params_df['target'].idxmax()].to_dict()
        params["window_size"] = int(np.around(params['window_size']))
        params["size_network"] = int(np.around(params['size_network']))
        params["population_size"] = int(np.around(params['population_size']))
        print('Max Params:\n', json.dumps(params, indent=4))
        
        self.db.add(AgentParams(**params, market=self.market))
        self.db.commit()
        self.bayes_params = params

    def train(self, iterations: int = 500) -> None:
        bayes_params = self.bayes_params | \
        self.db.query(
            AgentParams
        ).filter(
            AgentParams.market == self.market,
        ).order_by(
            AgentParams.target.desc()
        ).first().__dict__

        self.agent = Agent(self.df, **bayes_params, **self.config)
        self.agent.stategy.train(iterations)

    def simulate(self) -> Tuple[DataFrame, DataFrame]:
        self.buy_history, self.sell_history = self.agent.simulate()
        return self.buy_history, self.sell_history

    def plot(self) -> None:
        plt.figure(figsize = (20, 10))
        plt.plot(self.agent.env.close, label = 'true close', c = 'g')
        buys = self.buy_history["time_index"].astype(int).to_list()
        sells = self.sell_history["time_index"].astype(int).to_list()
        plt.plot(
            self.agent.env.close, 'X', label = 'predict buy', markevery = buys, c = 'b'
        )
        plt.plot(
            self.agent.env.close, 'o', label = 'predict sell', markevery = sells, c = 'r'
        )
        plt.legend()
        plt.show()
