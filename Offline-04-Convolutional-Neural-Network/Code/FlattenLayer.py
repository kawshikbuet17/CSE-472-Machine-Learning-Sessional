class FlattenLayer:
    def __init__(self):
        self.last_input_shape = None
    
    def forward(self, input):
        self.last_input_shape = input.shape
        return input.reshape(input.shape[0], -1)
    
    def backprop(self, grad_output, learning_rate):
        return grad_output.reshape(self.last_input_shape)