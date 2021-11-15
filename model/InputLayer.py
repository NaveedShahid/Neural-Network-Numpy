from model.Layer import Layer
import numpy as np

class InputLayer(Layer):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def forward_propagation(self, inputs):
        assert(len(inputs.shape) == len(self.input_shape) + 1)
        return inputs

    def backward_propagation(self, grads):
        return np.zeros_like(grads)

    def get_weights(self):
        return []

    def get_output_shape(self):
        return self.input_shape