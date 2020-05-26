import time
import numpy as np
import pandas as pd

class Deep_Evolution_Strategy:
    inputs = None

    def __init__(
        self,
        weights,
        bias,
        reward_function,
        population_size,
        sigma,
        learning_rate
    ) -> None:
        self.weights = weights
        self.bias = bias
        self.reward_function = reward_function
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate

    def extract(self, population, person):
        weights = []
        bias = []
        for index, layer in enumerate(population[0][person]):
            weights.append(self.weights[index] + self.sigma * layer)
        for index, layer in enumerate(population[1][person]):
            bias.append(self.bias[index] + self.sigma * layer)
        return weights, bias

    def spawn(self, weights, bias, population_size):
        pop_weights = []
        pop_bias = []
        for _ in range(population_size):
            person_weights = []
            person_bias = []
            for layer in weights:
                person_weights.append(np.random.randn(*layer.shape))
            pop_weights.append(person_weights)
            for layer in bias[:-1]:
                person_bias.append(np.random.randn(*layer.shape))
            person_bias.append(np.random.randn())
            pop_bias.append(person_bias)
        return [pop_weights, pop_bias]

    def evolve(self, rewards, population) -> None:
        for index, layer in enumerate(self.weights):
            change_rate = self.learning_rate / (self.population_size * self.sigma)
            adjustments = np.dot(np.array([p[index] for p in population[0]]).T, rewards).T
            self.weights[index] = layer + change_rate * adjustments
        for index, layer in enumerate(self.bias):
            change_rate = self.learning_rate / (self.population_size * self.sigma)
            adjustments = np.dot(np.array([p[index] for p in population[1]]).T, rewards).T
            self.bias[index] = layer + change_rate * adjustments

    def get_rewards(self, population):
        rewards = np.zeros(self.population_size)
        for person in range(self.population_size):
            weights, bias = self.extract(population, person)
            rewards[person] = self.reward_function(weights, bias)
        return (rewards - np.mean(rewards)) / np.std(rewards)

    def train(self, epoch: int = 100) -> None:
        for _ in range(epoch):
            population = self.spawn(self.weights, self.bias, self.population_size)
            rewards = self.get_rewards(population)
            self.evolve(rewards, population)