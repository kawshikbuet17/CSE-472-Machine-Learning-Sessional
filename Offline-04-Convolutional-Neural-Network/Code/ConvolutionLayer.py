import numpy as np
import math

class ConvolutionLayer:
    def __init__(self, num_of_filters, kernel_size, padding, stride=1):
        self.num_of_filters = num_of_filters
        self.kernel_size_h = kernel_size
        self.kernel_size_w = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = None
        self.biases = None
        self.input = None
 
    def forward(self, input):
        self.input = input
        batch_size, height, width, channels = input.shape
 
        output_height = int(math.floor(
            (height - self.kernel_size_h + 2 * self.padding)) / self.stride + 1)
        output_width = int(math.floor(
            (width - self.kernel_size_w + 2 * self.padding)) / self.stride + 1)
        output_shape = (batch_size, output_height,
                        output_width, self.num_of_filters)
 
        if self.weights is None:
 
            # initialize weights and biases
            # also include channels in the shape
            # do i have to also include the channel numbers when initializing the weights?
            self.weights = np.random.randn(self.num_of_filters, self.kernel_size_h,
                                           self.kernel_size_w, channels) / np.sqrt(self.kernel_size_h * self.kernel_size_w * channels)
 
        if self.biases is None:
            self.biases = np.zeros(self.num_of_filters)
 
        # pad the input data
        self.input_with_padding = np.pad(input, ((
            0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant')
 
 
        output = np.zeros(output_shape)
 
        for i in range(output_height):
            for j in range(output_width):
                input_matrix = self.input_with_padding[:, i * self.stride: i * self.stride +
                                                      self.kernel_size_h, j * self.stride: j * self.stride + self.kernel_size_w, :]
                input_matrix = input_matrix.reshape(batch_size, -1)
                # print("input matrix: ", input_matrix.shape)
                # print("weights: ", self.weights.shape)
                temp_weight = self.weights.reshape(self.num_of_filters, -1)
                # print("temp weight: ", temp_weight.shape)
                output[:, i, j, :] = np.dot(
                    input_matrix, temp_weight.T) + self.biases
 
        return output
 
    def backprop(self, output_error, learning_rate=0.05):
        lr = learning_rate
        batch_size, height, width, channels = self.input.shape
        output_height, output_width, output_channels = output_error.shape[1:]
 
        # print("output error shape: ", output_error.shape)
        # print("input shape: ", self.input_shape)
 
        # initialize the weight error and bias error
        weight_error = np.zeros(self.weights.shape)
        bias_error = np.zeros(self.biases.shape)
 
        # initialize the input error
        input_error = np.zeros(self.input.shape)
 
        # initialize the input error for padding
        input_error_padded = np.zeros(self.input_with_padding.shape)
 
        for i in range(output_height):
            for j in range(output_width):
                input_matrix = self.input_with_padding[:, i * self.stride: i * self.stride +
                                                      self.kernel_size_h, j * self.stride: j * self.stride + self.kernel_size_w, :]
                input_matrix = input_matrix.reshape(batch_size, -1)
                # print("input matrix: ", input_matrix.shape)
                # print("weights: ", self.weights.shape)
                temp_weight = self.weights.reshape(self.num_of_filters, -1)
                # print("temp weight: ", temp_weight.shape)
                weight_error += np.dot(output_error[:, i, j, :].T,
                                       input_matrix).reshape(self.weights.shape)
                bias_error += np.sum(output_error[:, i, j, :], axis=0)
 
                input_error_padded[:, i * self.stride: i * self.stride + self.kernel_size_h, j * self.stride: j * self.stride + self.kernel_size_w, :] += np.dot(
                    output_error[:, i, j, :], temp_weight).reshape(batch_size, self.kernel_size_h, self.kernel_size_w, channels)
 
        # remove the padding from the input error
        input_error = input_error_padded[:, self.padding: self.padding + height,
                                         self.padding: self.padding + width, :]
 
        # update the weights and biases
        self.weights = self.update_weight(weight_error, lr)
        self.biases = self.update_bias(bias_error, lr)
 
        return input_error
 
    def update_weight(self, weight_error, lr):
        temp_weight = self.weights - lr * weight_error
        return temp_weight
 
    def update_bias(self, bias_error, lr):
        temp_bias = self.biases - lr * bias_error
        return temp_bias