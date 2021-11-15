import abc 

class Layer(abc.ABC):
    def __init__(self):
        1+1

    @abc.abstractmethod
    def forward_propagation(self):
        return NotImplemented

    @abc.abstractmethod
    def backward_propagation(self, grads):
        return NotImplemented

    @abc.abstractmethod
    def get_weights(self):
        return NotImplemented

    @abc.abstractmethod
    def get_output_shape(self):
        return NotImplemented