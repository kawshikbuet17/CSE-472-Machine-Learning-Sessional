import numpy as np

class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X.copy()
        return np.maximum(0, X)

    def backprop(self, dout, learning_rate):
        dX = dout.copy()
        dX[self.X <= 0] = 0
        return dX