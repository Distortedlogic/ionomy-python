import numpy as np

class Model:
    def __init__(self, input_size: int, num_layers, layer_size, output_size: int):
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.weights = [
            np.random.randn(input_size, num_layers*layer_size),
            *[np.random.randn((i+1)*layer_size, i*layer_size) for i in range(1, num_layers)[::-1]],
            np.random.randn(layer_size, output_size),
            np.random.randn(layer_size, 1),
            np.random.randn(num_layers, layer_size),
        ]

    def build_feed(self, inputs) -> np.array:
        feed = np.dot(inputs, self.weights[0]) + self.weights[-1][0]
        for i in range(1, self.num_layers):
            feed = np.dot(feed, self.weights[i]) + self.weights[-1][i]
        return feed

    def predict(self, inputs):
        feed = self.build_feed(inputs)
        decision = np.dot(feed, self.weights[-3])
        buy = np.dot(feed, self.weights[-2])
        return decision, buy

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights
