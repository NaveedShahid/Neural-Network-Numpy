import numpy as np
from model.utils import get_smoothing

EPSILON = 1e-9

class BatchGradientDescent():
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def optimize(self, model):
        for layer in model.layers:
            for para, grad in layer.get_weights():
                para -= self.learning_rate * grad

class Adam():
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta1_exp = 1
        self.beta2_exp = 1
        self.caches_m = {}
        self.caches_v = {}

    def optimize(self, model):
        self.beta1_exp *= self.beta1
        self.beta2_exp *= self.beta2
        for layer in model.layers:
            parameters = layer.get_weights()
            cache_m = self.caches_m.get(layer, [None] * len(parameters))
            cache_v = self.caches_v.get(layer, [None] * len(parameters))
            for i, (para, grad) in enumerate(parameters):
                cache_m[i] = get_smoothing(cache_m[i], grad, self.beta1)
                cache_v[i] = get_smoothing(cache_v[i], grad ** 2, self.beta2)
                m_head = cache_m[i] / (1 - self.beta1_exp)
                v_head = cache_v[i] / (1 - self.beta2_exp)
                para -= self.learning_rate * m_head / (np.sqrt(v_head) + EPSILON)
