from model.Layer import Layer
import numpy as np

class SoftmaxLayer(Layer):
    def __call__(self, old_layer):
        self.output_shape = old_layer.get_output_shape()
        return self

    def forward_propagation(self, inputs):
        inputs_m = inputs - inputs.max(axis=-1, keepdims=True)
        self.old_outputs = np.exp(inputs_m) / np.exp(inputs_m).sum(axis=1)[:, None]
        return self.old_outputs

    def backward_propagation(self, grad):
        grad = self.old_outputs * (grad - (grad * self.old_outputs).sum(axis=1)[:, None])
        return grad

    def get_weights(self):
        return []

    def get_output_shape(self):
        return self.output_shape