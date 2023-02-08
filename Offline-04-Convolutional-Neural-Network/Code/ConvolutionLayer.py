import numpy as np
import math

class ConvolutionLayer:
    def __init__(self, num_of_filters, kernel_size, padding, stride=1):
        self.num_of_filters = num_of_filters
        self.kernel_size_h = kernel_size
        self.kernel_size_w = kernel_size
        self.stride = stride
        self.padding = padding
        self.W = None
        self.b = None
        self.X = None
 
    def forward(self, X):
        self.X = X.copy()
        batch_size, height, width, num_of_channels = X.shape

        height = (height - self.kernel_size_h + 2 * self.padding) / self.stride + 1
        height = int(math.floor(height))
        width = (width - self.kernel_size_w + 2 * self.padding) / self.stride + 1
        width = int(math.floor(width))

        Z = np.zeros([batch_size, height, width, self.num_of_filters])

        if self.W is None:
            # Xavier initialization
            self.W = np.random.randn(self.num_of_filters, self.kernel_size_h, self.kernel_size_w, num_of_channels)
            self.W = self.W * np.sqrt(2.0 / (self.kernel_size_h * self.kernel_size_w * num_of_channels))

        if self.b is None:
            self.b = np.zeros(self.num_of_filters)
 
        # pad the X data
        self.X_padded = np.pad(X, ((
            0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant')
 
        for sample in range(batch_size):
            for i in range(height):
                for j in range(width):
                    # determine the input slice
                    h_start = i * self.stride
                    h_end = i * self.stride + self.kernel_size_h
                    w_start = j * self.stride
                    w_end = j * self.stride + self.kernel_size_w

                    input_slice = self.X_padded[sample, h_start: h_end, w_start: w_end, :]
 
                    # perform the convolution with vectorization
                    Z[sample, i, j, :] = np.sum(input_slice * self.W, axis=(1, 2, 3)) + self.b
        return Z
 
    def backprop(self, dZ, learning_rate):
        batch_size, height, width, num_of_channels = self.X.shape
        Z_height, Z_width, Z_num_of_channels = dZ.shape[1:]

        # initialize the X error for padding
        dX_padded = np.zeros(self.X_padded.shape)

        # initialize the weight error and bias error
        dW = np.zeros(self.W.shape)
        db = np.zeros(self.b.shape)
 
        # initialize the X error
        dX = np.zeros(self.X.shape)

        for sample in range(batch_size):
            for i in range(Z_height):
                for j in range(Z_width):
                    # determine the input slice
                    h_start = i * self.stride
                    h_end = i * self.stride + self.kernel_size_h
                    w_start = j * self.stride
                    w_end = j * self.stride + self.kernel_size_w

                    input_slice = self.X_padded[sample, h_start: h_end, w_start: w_end, :]

                    # perform the convolution with vectorization
                    dW += dZ[sample, i, j, :, np.newaxis, np.newaxis, np.newaxis] * input_slice
                    db += dZ[sample, i, j, :]
                    dX_padded[sample, h_start: h_end, w_start: w_end, :] += np.sum(
                        dZ[sample, i, j, :, np.newaxis, np.newaxis, np.newaxis] * self.W, axis=0)
                    
        # remove the padding
        dX = dX_padded[:, self.padding: -self.padding, self.padding: -self.padding, :]

        # update the weights and bias
        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return dX
