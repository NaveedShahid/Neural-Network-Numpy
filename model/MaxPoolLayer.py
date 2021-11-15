from model.Layer import Layer
import numpy as np
from model.utils import im2col, col2im

class MaxPoolLayer(Layer):
    def __init__(self, kernel_size=[2, 2], stride=[2, 2]):
        self.kernel_size = kernel_size[0]
        self.stride = stride[0]

    def __call__(self, old_layer):
        pre_shape = old_layer.get_output_shape()
        self.output_depth = pre_shape[0]
        self.height = pre_shape[1]

        if self.stride == 1:
            self.padding = self.kernel_size - 1
        else:
            self.padding = (
                self.stride - (self.height - self.kernel_size) % self.stride)
            self.padding = self.padding % self.stride
        self.height_out = (
            (self.height - self.kernel_size + self.padding) // self.stride + 1)

        return self

    def forward_propagation(self, inputs):
        self.old_inputs = inputs
        batch_size = inputs.shape[0]

        inputs_reshaped = inputs.reshape(
            batch_size * self.output_depth, 1, self.height, self.height)
        self.inputs_v = im2col(
            inputs_reshaped, self.kernel_size, self.kernel_size, self.padding, self.stride)

        self.max_idx = np.argmax(self.inputs_v, axis=0)
        self.out = self.inputs_v[self.max_idx, range(self.max_idx.size)]

        self.out = self.out.reshape(
            self.height_out, self.height_out, batch_size, self.output_depth)
        self.out = self.out.transpose(2, 3, 0, 1)

        return self.out

    def backward_propagation(self, grads):
        batch_size = self.old_inputs.shape[0]

        inputs_view = np.zeros_like(self.inputs_v)
        dout_col = grads.transpose(2, 3, 0, 1).ravel()

        inputs_view[self.max_idx, range(dout_col.size)] = dout_col

        grads = col2im(
            inputs_view, (batch_size * self.output_depth, 1, self.height, self.height),
            self.kernel_size, self.kernel_size, padding=self.padding, stride=self.stride)
        grads = grads.reshape(self.old_inputs.shape)

        return grads

    def get_weights(self):
        return []

    def get_output_shape(self):
        return (self.output_depth, self.height_out, self.height_out)
