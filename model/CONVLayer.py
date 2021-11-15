import numpy as np

EPSILON = 1e-9

from model.utils import im2col, col2im
from model.Layer import Layer

class CONVLayer(Layer):
    def __init__(self, output_depth, kernel_size, stride=1):
        self.output_depth = output_depth
        self.kernel_size = kernel_size[0]
        self.stride = stride[0]

    def __call__(self, old_layer):
        pre_shape = old_layer.get_output_shape()
        depth, height, _ = pre_shape
        self.weights = np.random.uniform(
            low=-np.sqrt(6 / (depth + self.output_depth)),
            high=np.sqrt(6 / (depth + self.output_depth)),
            size=(self.output_depth, depth, self.kernel_size, self.kernel_size))
        self.bias = np.zeros((self.output_depth, 1))

        if self.stride == 1:
            self.padding = self.kernel_size - 1
        else:
            self.padding = self.stride - (height - self.kernel_size) % self.stride
            self.padding = self.padding % self.stride
        self.height_out = ((height - self.kernel_size + self.padding) // self.stride + 1)
        return self

    def forward_propagation(self, inputs):
        batch_num = inputs.shape[0]

        inputs_v = im2col(inputs, self.kernel_size, self.kernel_size, self.padding, self.stride)
        W_col = self.weights.reshape(self.output_depth, -1)
        out = W_col @ inputs_v + self.bias
        out = out.reshape(self.output_depth, self.height_out, self.height_out, batch_num)
        out = out.transpose(3, 0, 1, 2)

        self.inputs = inputs
        self.inputs_v = inputs_v

        return out

    def backward_propagation(self, grads):
        self.grad_bias = np.sum(grads, axis=(0, 2, 3))
        self.grad_bias = self.grad_bias.reshape(self.output_depth, -1)

        output_view = grads.transpose(1, 2, 3, 0).reshape(self.output_depth, -1)
        self.grad_weights = output_view @ self.inputs_v.T
        self.grad_weights = self.grad_weights.reshape(self.weights.shape)

        W_reshape = self.weights.reshape(self.output_depth, -1)
        inputs_view = W_reshape.T @ output_view
        grads = col2im(inputs_view, self.inputs.shape, self.kernel_size, self.kernel_size, padding=self.padding, stride=self.stride)

        return grads

    def get_weights(self):
        return [(self.weights, self.grad_weights), (self.bias, self.grad_bias)]

    def get_output_shape(self):
        return (self.output_depth, self.height_out, self.height_out)

