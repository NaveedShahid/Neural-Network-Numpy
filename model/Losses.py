import numpy as np
EPSILON = 1e-9
class CrossEntropy():
    def forward_propagation(self, pred, true):
        self.old_pred = pred.clip(min=EPSILON, max=None)
        self.old_truth = true
        loss = np.sum(np.where(self.old_truth == 1,
                               -np.log(self.old_pred), 0), axis=-1)
        loss = np.mean(loss)
        return loss

    def backward_propagation(self):
        return np.where(self.old_truth == 1, -1 / self.old_pred, 0)