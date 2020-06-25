from pandas.core.frame import DataFrame
import json

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

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
        env_config,
        tools,
        defaults,
        search_grid
    ) -> None:
        self.ohlcv_df = ohlcv_df
        self.env_config = env_config
        self.tools = tools
        self.defaults = defaults
        self.search_grid = search_grid
        self.params = {}
        self.bae_nn = BayesianOptimization(
            self.alpha_chad,
            self.search_grid,
            random_state = 42,
            verbose=2
        )
        self.bae_nn.subscribe(Events.OPTIMIZATION_STEP, JSONLogger(path="./logs.json"))
        try:
            load_logs(self.bae_nn, logs=["./logs.json"])
        except:
            pass
        
    def alpha_chad(self, **kwargs) -> float:
        params = {
            "window_size": int(np.around(kwargs.get('window_size', self.defaults['window_size']))),
            "network_size": int(np.around(kwargs.get('network_size', self.defaults['network_size']))),
            "population_size": int(np.around(kwargs.get('population_size', self.defaults['population_size']))),
            "tournsize": int(np.around(kwargs.get('tournsize', self.defaults['tournsize']))),
            "mu": kwargs.get('mu', self.defaults['mu']),
            "sigma": kwargs.get('sigma', self.defaults['sigma']),
            "indpb": kwargs.get('indpb', self.defaults['indpb']),
            "cxpb": kwargs.get('cxpb', self.defaults['cxpb']),
            "mutpb": kwargs.get('mutpb', self.defaults['mutpb']),
            "output_size": self.defaults['output_size']
        }
        env = Environment(self.ohlcv_df, window_size = params["window_size"], **self.env_config)
        chad_army = ChadArmy(**params, **self.tools, env=env)
        return chad_army.war(self.defaults['iterations'])

    def optimize(self, **kwargs) -> None:
        self.bae_nn.maximize(**kwargs)
        params = self.bae_nn.max["params"]
        params["window_size"] = int(np.around(params['window_size']))
        params["size_network"] = int(np.around(params['size_network']))
        params["population_size"] = int(np.around(params['population_size']))
        params["tournsize"] = int(np.around(params['tournsize']))
        print('Max Params:\n', json.dumps(params, indent=4))
        self.params = params