import numpy as np
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential


class Model:
    def __init__(self, input_dim, network_size, output_dim):
        self.model = Sequential()
        self.model.add(Dense(input_dim=input_dim+1, units=network_size, activation="relu"))
        self.model.add(Dense(input_dim=network_size, units=output_dim, activation="softmax"))
        self.model.compile(loss='mse', optimizer='rmsprop')

    def predict(self, state, position):
        inputs = np.array([np.concatenate([state, [position]])])
        return np.argmax(self.model.predict(inputs))

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
