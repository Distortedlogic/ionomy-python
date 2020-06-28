import numpy as np
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


class Model:
    def __init__(self, input_shape, output_dim):
        # self.model = Sequential()
        # self.model.add(Dense(input_dim=input_dim+1, units=network_size, activation="relu"))
        # self.model.add(Dense(input_dim=network_size, units=output_dim, activation="softmax"))
        # self.model.compile(loss='mse', optimizer='rmsprop')

        self.model = Sequential()
        self.model.add(Conv2D(
            1,
            kernel_size=(input_shape[1], input_shape[1]),
            strides=(input_shape[1] + 1, 1),
            activation='relu',
            input_shape=input_shape
        ))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(output_dim, activation='softmax'))
        self.model.compile(loss='mse', optimizer='rmsprop')

    def predict(self, state):
        return np.argmax(self.model.predict(np.expand_dims(np.expand_dims(state, axis=2), axis=0)))

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
