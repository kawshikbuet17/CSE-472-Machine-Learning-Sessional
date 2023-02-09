import numpy as np

class MaxPoolingLayer:
 
    def __init__(self, pool_size , stride=2):
        self.pool_size = (pool_size, pool_size)
        self.stride = stride
 
    def forward(self, X):
        self.input_shape = X.shape
        self.X = X.copy()
        batch_size, height, width, channels = X.shape
 
        Z_height = int((height - self.pool_size[0]) / self.stride + 1)
        Z_width = int((width - self.pool_size[1]) / self.stride + 1)
        Z_shape = (batch_size, Z_height, Z_width, channels)

        Z = np.zeros(Z_shape)
 
        for b in range(batch_size):
            h_indices = np.arange(0, height - self.pool_size[0] + 1, self.stride)
            w_indices = np.arange(0, width - self.pool_size[1] + 1, self.stride)
 
            for i in h_indices:
                for j in w_indices:
                    h_end = i + self.pool_size[0]
                    w_end = j + self.pool_size[1]
                    current_region = X[b, i:h_end, j:w_end, :]
 
                    Z[b, int(i/self.stride), int(j/self.stride),:] = np.max(current_region, axis=(0, 1))

        return Z
 
 
 
    def backprop(self, dZ, learning_rate):
        batch_size, height, width, channels = dZ.shape
        dX = np.zeros(self.input_shape)
 
        stride = self.stride
        pool_size = self.pool_size
 
        for b in range(batch_size):
            for m in range(channels):
                for i in range(0, height - pool_size[0] + 1, stride):
                    for j in range(0, width - pool_size[1] + 1, stride):
                        h_end = i + pool_size[0]
                        w_end = j + pool_size[1]
                        current_region = self.X[b, i:h_end, j:w_end, m]

                        max_index = np.argmax(current_region)
                        k, l = np.unravel_index(max_index, pool_size)
                        dX[b, i+k, j+l, m] = dZ[b, int(i/stride), int(j/stride), m]
 
        return dX