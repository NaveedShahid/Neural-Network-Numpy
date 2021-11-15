from model.Layer import Layer
import numpy as np

class DropoutLayer(Layer):
    def __init__(self, dropout_rate):
        self.dropout_keep_rate = 1 - dropout_rate

    def __call__(self, old_layer):
        self.output_shape = old_layer.get_output_shape()
        return self

    def forward_propagation(self, inputs):
        self.mask = np.random.binomial(
            1, self.dropout_keep_rate, inputs.shape[1:])
        return inputs * self.mask / self.dropout_keep_rate

    def backward_propagation(self, grads):
        return grads * self.mask / self.dropout_keep_rate

    def get_weights(self):
        return []

    def get_output_shape(self):
        return self.output_shape
