import json

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from pandas.core.frame import DataFrame

from .chad_army import ChadArmy
from .environment import Environment

# from .schema import AgentParams, Base
# engine = create_engine('sqlite:///database.sqlite')
# Session = sessionmaker(bind=engine)
# Base.metadata.create_all(engine)

class Bae:
    def __init__(
        self,
        ohlcv_df: DataFrame,
        iterations: int,
        initial_capital: int,
        max_buy: float,
        max_sell: float,
        output_size: int,
        toolbox,
        stats
    ) -> None:
        self.ohlcv_df = ohlcv_df
        self.iterations = iterations
        self.chad_config = {
            "output_size": output_size,
            "toolbox": toolbox,
            "stats": stats
        }
        self.env_config = {
            "initial_capital": initial_capital,
            "max_buy": max_buy,
            "max_sell": max_sell
        }
        
    def alpha_chad(
        self,
        window_size: int,
        network_size: int,
        population_size: int,
        tournsize: int,
        mu: float,
        sigma: float,
        indpb: float,
        cxpb: float,
        mutpb: float
    ):
        params = {
            "window_size": int(np.around(window_size)),
            "network_size": int(np.around(network_size)),
            "population_size": int(np.around(population_size)),
            "tournsize": int(np.around(tournsize)),
            "mu": mu,
            "sigma": sigma,
            "indpb": indpb,
            "cxpb": cxpb,
            "mutpb": mutpb
        }
        env = Environment(self.ohlcv_df, params["window_size"], **self.env_config)
        chad_army = ChadArmy(**params, env=env, **self.chad_config)
        return chad_army.war(self.iterations)

    def optimize(
        self,
        init_points: int = 30,
        n_iter: int = 50,
        acq: str = 'ei',
        xi: float = 0.0
    ) -> None:
        NN_BAYESIAN = BayesianOptimization(
            self.alpha_chad,
            {
                'window_size': (2, 50),
                "network_size": (10,1000),
                'population_size': (5, 100),
                "tournsize": (3,20),
                "mu": (-1, 1),
                'sigma': (0.01, 0.99),
                "indpb": (0.01, 0.25),
                "cxpb": (0, 1),
                "mutpb": (0, 1)
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
        params["tournsize"] = int(np.around(params['tournsize']))
        print('Max Params:\n', json.dumps(params, indent=4))
        self.params = params
        return params