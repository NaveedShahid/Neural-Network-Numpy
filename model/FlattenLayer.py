from model.Layer import Layer

# inherit from base class Layer
class FlattenLayer(Layer):
    def __call__(self, old_layer):
        pre_shape = old_layer.get_output_shape()
        self.output_units = 1
        for dim in pre_shape:
            self.output_units *= dim
        return self

    def forward_propagation(self, inputs):
        self.old_inputs = inputs
        inputs = inputs.reshape((inputs.shape[0], -1))
        return inputs

    def backward_propagation(self, grads):
        grads = grads.reshape(self.old_inputs.shape)
        return grads

    def get_weights(self):
        return []

    def get_output_shape(self):
        return (self.output_units,)