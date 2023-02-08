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
        self.X = X
        batch_size, height, width, num_of_channels = X.shape

        # calculate dZ shape
        height = (height - self.kernel_size_h + 2 * self.padding) / self.stride + 1
        height = int(math.floor(height))
        width = (width - self.kernel_size_w + 2 * self.padding) / self.stride + 1
        width = int(math.floor(width))

        Z = np.zeros((batch_size, 
                          height, 
                          width, 
                          self.num_of_filters))

        if self.W is None:
            # xaiver initialization
            self.W = np.random.randn(self.num_of_filters, self.kernel_size_h, self.kernel_size_w, num_of_channels)
            self.W = self.W * np.sqrt(2.0 / (self.kernel_size_h * self.kernel_size_w * num_of_channels))
 
        if self.b is None:
            self.b = np.zeros(self.num_of_filters)

        self.X = np.pad(X, ((0, 0), 
            (self.padding, self.padding), 
            (self.padding, self.padding), 
            (0, 0)), 
            'constant'
            )

        for sample in range(batch_size):
            for i in range(height):
                for j in range(width):
                    # determince the input slice
                    h_start = i * self.stride
                    h_end = i * self.stride + self.kernel_size_h
                    w_start = j * self.stride
                    w_end = j * self.stride + self.kernel_size_w

                    input_slice = self.X[sample, h_start: h_end, w_start: w_end, :]
                    
                    # perform the convolution with vectorization
                    Z[sample, i, j, :] = np.sum(input_slice * self.W, axis=(1, 2, 3)) + self.b
        return Z
 
    def backprop(self, dZ, learning_rate):
        batch_size, height, width, channels = self.X.shape
        dZ_height, dZ_width, dZ_channels = dZ.shape[1:]

        dW = np.zeros(self.W.shape)
        db = np.zeros(self.b.shape)
        dX = np.zeros(self.X.shape)
 
        # initialize the input error for padding
        dX_padded = np.zeros(self.X.shape)

        # calculate the gradient of the weights
        for sample in range(batch_size):
            for i in range(dZ_height):
                for j in range(dZ_width):
                    # determine the input slice
                    h_start = i * self.stride
                    h_end = i * self.stride + self.kernel_size_h
                    w_start = j * self.stride
                    w_end = j * self.stride + self.kernel_size_w
                    input_slice = self.X[sample, h_start: h_end, w_start: w_end, :]
                    
                    # calculate the gradient of the weights
                    dW += np.dot(dZ[sample, i, j, :].T, input_slice).reshape(self.W.shape)
                    db += np.sum(dZ[sample, i, j, :], axis=0)

                    # calculate the gradient of the input
                    dX_padded[sample, h_start: h_end, w_start: w_end, :] += np.dot(self.W, dZ[sample, i, j, :].T).reshape(input_slice.shape)

        # remove the padding from the input error
        dX = dX_padded[:, self.padding: self.padding + height,
                                         self.padding: self.padding + width, :]
 
        self.W = self.W - learning_rate * dW
        self.b = self.b - learning_rate * db
 
        return dX