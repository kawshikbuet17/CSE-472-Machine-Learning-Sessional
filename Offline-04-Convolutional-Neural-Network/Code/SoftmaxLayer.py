import numpy as np

class SoftmaxLayer:
    def __init__(self):
        self.last_input = None
    
    def forward(self, input):
        self.last_input = input
        exp = np.exp(input - np.max(input, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)
    
    def backprop(self, grad_output, learning_rate):
        return grad_output