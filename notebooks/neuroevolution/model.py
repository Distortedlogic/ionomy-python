from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import SimpleRNN
import numpy as np

class Model:
    def __init__(self, input_dim, network_size, output_dim):
        self.model = Sequential()
        self.model.add(SimpleRNN(
            batch_input_shape=(1, 1, input_dim),
            output_dim=network_size,
            stateful=True,
            activation="relu"
        ))
        self.model.add(Dense(input_dim=network_size, output_dim=output_dim, activation="softmax"))
        self.model.compile(loss='mse', optimizer='rmsprop')

    def predict(self, inputs):
        return self.model.predict(inputs)

    def flatten(self):
        return np.concatenate([layer.flatten() for layer in self.model.get_weights()])

    def set_weights(self, new_weights):
        accum = 0
        for layer in self.model.layers:
            current_layer_weights_list = layer.get_weights()
            new_layer_weights_list = []
            for layer_weights in current_layer_weights_list:
                layer_total = np.prod(layer_weights.shape)
                new_layer_weights_list.append(
                    new_weights[accum:accum + layer_total].
                        reshape(layer_weights.shape))
                accum += layer_total
            layer.set_weights(new_layer_weights_list)
