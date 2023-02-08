import numpy as np

class ReLULayer:
    def __init__(self):
        self.last_input = None

    def forward(self, input):
        self.last_input = input
        return np.maximum(0, input)

    def backprop(self, d_L_d_out, learning_rate):
        d_L_d_input = d_L_d_out.copy()
        d_L_d_input[self.last_input <= 0] = 0
        return d_L_d_input