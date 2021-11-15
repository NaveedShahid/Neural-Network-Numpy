from model.utils import get_smoothing
from model.Layer import Layer
import numpy as np

EPSILON=1e-9

class BatchNormLayer(Layer):
    def __init__(self, momentum=0.95):
        self.momentum = momentum

    def __call__(self, old_layer):
        self.output_shape = old_layer.get_output_shape()
        self.gamma = np.random.uniform(-0.1, 0.1, self.output_shape)
        self.beta = np.zeros(self.output_shape)
        self.running_mean = 0
        self.running_var = 0
        return self

    def forward_propagation(self, inputs):
        self.old_inputs = inputs
        self.old_mu = np.mean(inputs, axis=0)
        self.old_var = np.var(inputs, axis=0)

        self.inputs_norm = (inputs - self.old_mu) / np.sqrt(self.old_var + EPSILON)
        inputs = self.gamma * self.inputs_norm + self.beta

        self.running_mean = get_smoothing(
            self.running_mean, self.old_mu, self.momentum)
        self.running_var = get_smoothing(
            self.running_var, self.old_var, self.momentum)

        return inputs

    def backward_propagation(self, grad):
        m, _ = self.old_inputs.shape

        inputs_mu = self.old_inputs - self.old_mu
        std_inv = 1. / np.sqrt(self.old_var + EPSILON)

        grad_norm = grad * self.gamma
        dvar = np.sum(grad_norm * inputs_mu, axis=0) * -.5 * std_inv ** 3
        dmu = (np.sum(grad_norm * - std_inv, axis=0) +
               dvar * np.mean(-2. * inputs_mu, axis=0))

        grad = (grad_norm * std_inv) + (dvar * 2 * inputs_mu / m) + (dmu / m)
        self.grad_gamma = np.sum(grad * self.inputs_norm, axis=0)
        self.grad_beta = np.sum(grad, axis=0)

        return grad

    def get_weights(self):
        return [(self.gamma, self.grad_gamma), (self.beta, self.grad_beta)]

    def get_output_shape(self):
        return self.output_shape
