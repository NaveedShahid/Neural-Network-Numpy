import numpy as np
import time
from model.utils import generator, Dataset

class Network():
    def __init__(self, layers=[]):
        self.layers = []
        for layer in layers:
            self.add(layer)

    def add(self, layer):
        if len(self.layers) != 0:
            layer = layer(self.layers[-1])            
        self.layers.append(layer)

    def use(self, loss_fn, opt, metric='acc'):
        self.loss_fn = loss_fn
        self.opt = opt
        self.metrics = metric

    def train(self, x_mini, y_mini):
        y_pred = self.forward_propagation(x_mini)
        loss = self.loss_function(y_mini, y_pred)
        self.backward_propagation_propagation()
        return [loss, self.get_accuracy(y_mini, y_pred)]

    def forward_propagation(self, inputs):
        for layer in self.layers:
            inputs = layer.forward_propagation(inputs)
        return inputs

    def backward_propagation_propagation(self):
        grads = self.loss_fn.backward_propagation()
        for i in range(len(self.layers)-1, -1, -1):
            grads = self.layers[i].backward_propagation(grads)
        self.opt.optimize(self)

    def loss_function(self, y_true, y_pred):
        return self.loss_fn.forward_propagation(y_pred, y_true)

    def fit(self, x, y, epochs=100, batch_size=32):
        train = Dataset(X=x, y=y, batch_size=batch_size)
        for i in range(epochs):
            start = time.time()
            print("\n====== Epoch", i, "======")            
            losses = 0
            accs = 0 
            
            for batch, (x_mini, y_mini) in enumerate(train): 
                batch=batch+1
                loss,acc = self.train(x_mini, y_mini)
                losses += loss
                accs += acc
                print("Loss: ",loss,"\tAcc: ",acc)
                
            print("\nEpoch Loss: ",losses/batch,"\tAcc: ",accs/batch,'\n')
            print("Time: ", time.time()-start)
        return self

    def predict(self, X, batch_size=32):
        y_pred = []
        for x in generator(X, batch_size, shuffle=False):
            y_ = self.forward_propagation(x)
            y_pred.append(y_)
        y_pred = np.concatenate(y_pred, axis=0)
        return y_pred

    def get_accuracy(self, y_true, y_pred):
        y_true = np.argmax(y_true,axis=-1)
        y_pred = np.argmax(y_pred,axis=-1)
        accuracy = np.sum(y_true==y_pred)/len(y_true)
        return np.array(accuracy)