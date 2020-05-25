import numpy as np

class Model:
    def __init__(self, input_size: int, num_layers: int, layer_size: int, output_size: int):
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.weights = [
            np.random.randn(input_size, num_layers*layer_size),
            *[np.random.randn((i+1)*layer_size, i*layer_size) for i in range(1, num_layers)[::-1]],
            np.random.randn(layer_size, output_size),
            np.random.randn(layer_size, 1)
        ]
        self.bias = [
            *[np.random.randn((i+1)*layer_size) for i in range(num_layers)[::-1]],
            np.random.randn(output_size),
            np.random.randn()
        ]

    def build_feed(self, inputs) -> np.array:
        feed = np.dot(inputs, self.weights[0]) + self.bias[0]
        for i in range(1, self.num_layers):
            feed = np.dot(feed, self.weights[i]) + self.bias[i]
        return feed

    def predict(self, inputs):
        feed = self.build_feed(inputs)
        decision = np.dot(feed, self.weights[-2]) + self.bias[-2]
        buy = np.dot(feed, self.weights[-1]) + self.bias[-1]
        return decision, buy
