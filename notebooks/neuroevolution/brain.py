import numpy as np
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, add
from keras.layers import Conv2D, MaxPooling2D, Concatenate, Input


class Brain:
    def __init__(self, input_shape, output_dim):
        input_size = input_shape[0] * input_shape[1] + 1
        state_input = Input(
            shape=input_size,
            name="state"
        )
        hidden = Dense(input_size // 10, activation='relu')(state_input)
        action_pred = Dense(output_dim, activation='softmax')(hidden)
        self.model = Model(inputs = [state_input], outputs = [action_pred])
        self.model.compile(loss='mse', optimizer='rmsprop')

    def predict(self, state, position):
        inputs = np.expand_dims(np.concatenate([state.flatten(), np.array([position])]), axis=0)
        return np.argmax(self.model.predict(inputs))

    def size(self):
        accum = 0
        for layer in self.model.layers:
            for layer_weights in layer.get_weights():
                accum += np.prod(layer_weights.shape)
        return accum

    def set_weights(self, new_weights):
        accum = 0
        for layer in self.model.layers:
            current_layer_weights_list = layer.get_weights()
            new_layer_weights_list = []
            for layer_weights in current_layer_weights_list:
                layer_total = np.prod(layer_weights.shape)
                new_layer_weights_list.append(
                    new_weights[accum:accum + layer_total].reshape(layer_weights.shape)
                )
                accum += layer_total
            layer.set_weights(new_layer_weights_list)
