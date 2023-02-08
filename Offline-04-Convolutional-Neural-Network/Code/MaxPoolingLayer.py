import numpy as np

class MaxPoolingLayer:
 
    def __init__(self, pool_size , stride=2):
        pool_size = (pool_size, pool_size)
        self.pool_size = pool_size
 
        self.stride = stride
 
    def forward(self, input_data):
 
        self.input_shape = input_data.shape
 
        self.input_data = input_data.copy()
 
        batch_size, height, width, channels = input_data.shape
 
 
 
        output_height = int((height - self.pool_size[0]) / self.stride + 1)
 
        output_width = int((width - self.pool_size[1]) / self.stride + 1)
 
        output_shape = (batch_size, output_height, output_width, channels)
 
        output = np.zeros(output_shape)
 
        for b in range(batch_size):
 
            h_indices = np.arange(
                0, height - self.pool_size[0] + 1, self.stride)
 
            w_indices = np.arange(
                0, width - self.pool_size[1] + 1, self.stride)
 
            for i in h_indices:
 
                for j in w_indices:
 
 
                    current_region = input_data[b, i:i +
                                                self.pool_size[0], j:j+self.pool_size[1], :]
 
                    output[b, int(i/self.stride), int(j/self.stride),
                           :] = np.max(current_region, axis=(0, 1))
 
 
        return output
 
 
 
    def backprop(self, output_error, learning_rate):
 
        batch_size, height, width, channels = output_error.shape
 
    # initialize the gradient of the input data
 
        input_data_gradient = np.zeros(self.input_shape)
 
        stride = self.stride
        pool_size = self.pool_size
 
        for b in range(batch_size):
 
            for m in range(channels):
 
                for i in range(0, height - pool_size[0] + 1, stride):
 
                    for j in range(0, width - pool_size[1] + 1, stride):
 
                        current_region = self.input_data[b, i:i + pool_size[0], j:j+pool_size[1], m]
 
                        max_value = np.max(current_region)
                        max_index = np.argmax(current_region)
 
                        k, l = np.unravel_index(max_index, pool_size)
 
                        input_data_gradient[b, i+k, j+l, m] = output_error[b, int(i/stride), int(j/stride), m]
 
 
        return input_data_gradient