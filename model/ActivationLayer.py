from model.Layer import Layer
import numpy as np

class ReluLayer(Layer):
    def __call__(self, old_layer):
        self.output_shape = old_layer.get_output_shape()
        return self

    def forward_propagation(self, inputs):
        self.old_inputs = inputs
        return np.clip(inputs, 0, None)

    def backward_propagation(self, grads):
        return np.where(self.old_inputs > 0, grads, 0)

    def get_weights(self):
        return []

    def get_output_shape(self):
        return self.output_shape

class SigmoidLayer(Layer):
    def __call__(self, old_layer):
        self.output_shape = old_layer.get_output_shape()
        return self

    def forward_propagation(self, inputs):
        self.old_outputs = np.exp(inputs) / (1. + np.exp(inputs))
        return self.old_outputs

    def backward_propagation(self, grads):
        return self.old_outputs * (1. - self.old_outputs) * grads

    def get_weights(self):
        return []

    def get_output_shape(self):
        return self.output_shape