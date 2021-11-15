from model.Layer import Layer
import numpy as np

class FeedForwardLayer(Layer):
    def __init__(self, output_units):
        self.output_units = output_units

    def __call__(self, old_layer):
        pre_shape = old_layer.get_output_shape()
        assert(len(pre_shape) == 1)
        self.input_units = pre_shape[0]
        self.W = np.random.rand(self.input_units, self.output_units) * (0.06) - (
            0.03)
        return self

    def forward_propagation(self, x):
        self.old_x = x
        return np.dot(x, self.W)

    def backward_propagation(self, grad):
        self.grad_W = np.mean(np.matmul(self.old_x[:, :, None], grad[:, None, :]), axis=0)
        return np.dot(grad, self.W.transpose())

    def get_weights(self):
        return [(self.W, self.grad_W)]

    def get_output_shape(self):
        return (self.output_units,)

class BiasLayer(Layer):
    def __init__(self):
        1+1
    def __call__(self, old_layer):
        self.output_shape = old_layer.get_output_shape()
        self.bias = np.zeros(self.output_shape)
        return self

    def forward_propagation(self, x):
        self.old_x = x
        return x + self.bias

    def backward_propagation(self, grad):
        self.grad_bias = grad.mean(axis=0)
        return grad

    def get_weights(self):
        return [(self.bias, self.grad_bias)]

    def get_output_shape(self):
        return self.output_shape

class FCLayer(Layer):
    def __init__(self, output_units):
        self.feedforward = FeedForwardLayer(output_units)
        self.bias = BiasLayer()

    def __call__(self, old_layer):
        old_layer = self.feedforward(old_layer)
        old_layer = self.bias(old_layer)
        return self

    def forward_propagation(self, x):
        x = self.feedforward.forward_propagation(x)
        x = self.bias.forward_propagation(x)
        return x

    def backward_propagation(self, grad):
        grad = self.bias.backward_propagation(grad)
        grad = self.feedforward.backward_propagation(grad)
        return grad

    def get_weights(self):
        return self.feedforward.get_weights() + self.bias.get_weights()

    def get_output_shape(self):
        return self.bias.get_output_shape()