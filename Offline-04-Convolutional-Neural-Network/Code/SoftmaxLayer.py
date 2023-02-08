import numpy as np

class SoftmaxLayer:
    def __init__(self):
        self.X = None
    
    def forward(self, X):
        self.X = X
        exp = np.exp(X - np.max(X, axis=1, keepdims=True))
        
        return exp / np.sum(exp, axis=1, keepdims=True)
    
    def backprop(self, dout, learning_rate):
        return dout