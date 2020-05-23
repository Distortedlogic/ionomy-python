import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization

from .agent import Agent
from .model import Model

from sqlalchemy import create_engine, and_, func
from sqlalchemy.orm.session import sessionmaker

from .schema import Base, AgentParams

engine = create_engine('sqlite:///database.sqlite')
Session = sessionmaker(bind=engine)
Base.metadata.create_all(engine)

class Bayes:
    def __init__(self, df, max_buy, max_sell, initial_capital, market) -> None:
        self.df = df
        self.initial_capital = initial_capital
        self.max_buy = max_buy
        self.max_sell = max_sell
        self.db = Session()
        self.market = market
        self.skip=2
    def agent(
        self,
        window_size: int,
        skip: int,
        population_size: int,
        sigma: float,
        learning_rate: float,
        size_network: int
    ):
        model = Model(window_size, size_network, 3)
        agent = Agent(
            self.df,
            population_size,
            sigma,
            learning_rate,
            model,
            self.initial_capital,
            self.max_buy,
            self.max_sell,
            self.skip,
            window_size,
        )
        try:
            agent.fit(100, 1000)
            return agent.es.reward_function(agent.es.weights)
        except:
            return 0

    def find_agent(
        self,
        window_size,
        population_size,
        sigma,
        learning_rate,
        size_network
    ):
        param = {
            'window_size': int(np.around(window_size)),
            'skip': self.skip,
            'population_size': int(np.around(population_size)),
            'sigma': max(min(sigma, 1), 0.0001),
            'learning_rate': max(min(learning_rate, 0.5), 0.000001),
            'size_network': int(np.around(size_network)),
        }
        return self.agent(**param)
    def bayes_maximize(self, init_points = 30, n_iter = 50, acq = 'ei', xi = 0.0):
        self.NN_BAYESIAN = BayesianOptimization(
            self.find_agent,
            {
                'window_size': (2, 50),
                'population_size': (1, 50),
                'sigma': (0.01, 0.99),
                'learning_rate': (0.000001, 0.49),
                'size_network': (10, 1000),
            },
            verbose=1
        )
        self.NN_BAYESIAN.maximize(init_points = init_points, n_iter = n_iter, acq = acq, xi = xi)
        params_df = pd.DataFrame.from_records([
            {"target": item["target"], **item["params"]} for item in self.NN_BAYESIAN.res
        ])
        params = params_df.iloc[params_df['target'].idxmax()].to_dict()
        params["window_size"] = int(np.around(params['window_size']))
        params["size_network"] = int(np.around(params['size_network']))
        params["population_size"] = int(np.around(params['population_size']))
        print('Max Params:\n', json.dumps(params, indent=4))
        self.db.add(AgentParams(**params, market=self.market))
        self.db.commit()
        self.max_params = params

    def run(self):
        params = \
        self.db.query(
            AgentParams
        ).filter(
            AgentParams.market == self.market,
        ).order_by(
            AgentParams.target.desc()
        ).first().__dict__

        model = Model(
            input_size = params['window_size'], 
            layer_size = params['size_network'], 
            output_size = 3
        )
        agent = Agent(
            self.df,
            population_size = params['population_size'], 
            sigma = params['sigma'], 
            learning_rate = params['learning_rate'], 
            model = model, 
            money = self.initial_capital, 
            max_buy = self.max_buy, 
            max_sell = self.max_sell, 
            skip = self.skip, 
            window_size = params['window_size']
        )
        agent.fit(500, 100)
        self.buy_history, self.sell_history, actions = agent.buy()
        return self.buy_history, self.sell_history, actions

    def plot(self):
        plt.figure(figsize = (20, 10))
        plt.plot(self.close, label = 'true close', c = 'g')
        buys = self.buy_history["time_index"].astype(int).to_list()
        sells = self.sell_history["time_index"].astype(int).to_list()
        plt.plot(
            self.close, 'X', label = 'predict buy', markevery = buys, c = 'b'
        )
        plt.plot(
            self.close, 'o', label = 'predict sell', markevery = sells, c = 'r'
        )
        plt.legend()
        plt.show()