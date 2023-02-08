import numpy as np

class FullyConnectedLayer:
    def __init__(self, output_size):
        self.weights = None
        self.bias = None
        self.output_size = output_size
        self.input = None
    
    def forward(self, input):
        # initialize weights and bias
        if self.weights is None:
            self.weights = np.random.randn(input.shape[1], self.output_size)
        if self.bias is None:
            self.bias = np.random.randn(self.output_size)

        self.input = input.copy()

        return np.dot(input, self.weights) + self.bias

    def backprop(self, grad_output, learning_rate):
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0)

        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias

        return grad_input