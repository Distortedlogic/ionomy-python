import time
import numpy as np
import pandas as pd

class Deep_Evolution_Strategy:
    inputs = None

    def __init__(
        self,
        weights,
        reward_function,
        population_size,
        sigma,
        learning_rate
    ) -> None:
        self.weights = weights
        self.reward_function = reward_function
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate

    def _get_weight_from_population(self, weights, population) -> list:
        return [
            weights[index] + self.sigma * p for index, p in enumerate(population)
        ]

    def get_weights(self):
        return self.weights

    def spawn_population(self, weights, population_size):
        return [
            [
                np.random.randn(*layer.shape) for layer in weights
            ] for _ in range(population_size)
        ]

    def train(self, epoch: int = 100):
        history = []
        for i in range(epoch):
            start = time.time()
            population = self.spawn_population(self.weights, self.population_size)
            rewards = np.zeros(self.population_size)
            for k in range(self.population_size):
                weights_population = self._get_weight_from_population(self.weights, population[k])
                rewards[k] = self.reward_function(weights_population)
            rewards = (rewards - np.mean(rewards)) / np.std(rewards)
            for index, layer in enumerate(self.weights):
                change_rate = self.learning_rate / (self.population_size * self.sigma)
                adjustments = np.dot(np.array([p[index] for p in population]).T, rewards).T
                self.weights[index] = layer + change_rate * adjustments
            history.append({
                "iteration": i+1,
                "train_time": time.time()-start,
                "reward": self.reward_function(self.weights)
            })
        return pd.DataFrame.from_records(history)