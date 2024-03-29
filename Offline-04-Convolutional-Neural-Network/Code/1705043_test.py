import pickle
import argparse
import cv2
import pandas as pd
import os
import numpy as np

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import math

import numpy as np
import os
import cv2
import pandas as pd

def load_data(image_folder, label_path, limit=None):
    images = []
    for filename in os.listdir(image_folder):
        img = cv2.imread(os.path.join(image_folder, filename))
        if img is not None:
            # convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # resize to 28x28
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
            # convert to float32
            img = img.astype(np.float32)
            images.append(img)
            if limit is not None and len(images) == limit:
                break

    df = pd.read_csv(label_path)
    labels = df['digit'].values
    labels = labels[:len(images)]

    return images, labels

def preprocess_data(images, labels):
    # detection friendly preprocessing
    for i in range(len(images)):
        images[i] = cv2.dilate(images[i], (3, 3))
        # images[i] = cv2.threshold(images[i], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # images[i] = cv2.GaussianBlur(images[i], (3, 3), 0)
        # images[i] = cv2.threshold(images[i], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # convert to numpy array
    images = np.array(images)/255
    labels = np.array(labels)

    # reshape images to 28x28x1
    images = images.reshape(images.shape[0], 28, 28, 1)
    # normalize images with std and mean
    images = (images - np.mean(images)) / np.std(images)

    return images, labels

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
    
    def destroy(self):
        self.X = None

class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X.copy()
        return np.maximum(0, X)

    def backprop(self, dout, learning_rate):
        dX = dout.copy()
        dX[self.X <= 0] = 0
        return dX
    
    def destroy(self):
        self.X = None

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

    def destroy(self):
        # do nothing
        pass
class FlattenLayer:
    def __init__(self):
        self.last_input_shape = None
    
    def forward(self, input):
        self.last_input_shape = input.shape
        return input.reshape(input.shape[0], -1)
    
    def backprop(self, grad_output, learning_rate):
        return grad_output.reshape(self.last_input_shape)

    def destroy(self):
        # do nothing
        pass

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
    def destroy(self):
        self.input = None

import numpy as np

class SoftmaxLayer:
    def __init__(self):
        self.X = None
    
    def forward(self, X):
        self.X = X.copy()
        exp = np.exp(X - np.max(X, axis=1, keepdims=True))
        
        return exp / np.sum(exp, axis=1, keepdims=True)
    
    def backprop(self, dout, learning_rate):
        return dout
    
    def destroy(self):
        self.X = None

import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
class Model:
    def __init__(self, num_classes):
        self.layers = []
        self.num_classes = num_classes

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def one_hot_encode(self, labels):
        one_hot = np.zeros((len(labels), self.num_classes))
        for i, label in enumerate(labels):
            one_hot[i][label] = 1
        return one_hot
    
    def cross_entropy(self, y_true, y_pred):
        eps = 1e-9
        return -np.sum(y_true * np.log(y_pred + eps))

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        print("y_pred (first 10)\t: ", y_pred[:10])
        print("y_real (first 10)\t: ", y[:10])
        accuracy = np.mean(y_pred == y)

        # calculate f1 score using sklearn
        f1 = f1_score(y, y_pred, average='macro')

        # calculate loss = summation of difference between y_pred and y
        loss = 0
        for i in range(len(y)):
            loss += np.sum(np.abs(y_pred[i] - y[i]))
        return loss, accuracy, f1 

    def train(self, X_train, y_train, X_val, y_val, learning_rate, epochs, batch_size):
        train_loss_history = []
        train_acc_history = []
        train_f1_history = []
        val_loss_history = []
        val_acc_history = []
        val_f1_history = []
        for epoch in range(epochs):
            print("epoch: ", epoch)

            # split data into batches
            batches = []
            for i in range(0, len(X_train), batch_size):
                batches.append((X_train[i:i+batch_size], y_train[i:i+batch_size]))

            # train model
            for i in range(len(batches)):
                X_batch, y_batch = batches[i]
                # print("\tbatch: ", i)
                y_batch_one_hot = self.one_hot_encode(y_batch)
                y_pred = self.forward(X_batch)
                loss = self.cross_entropy(y_batch_one_hot, y_pred)
                # print("\t\tloss: ", loss)
                grad = y_pred - y_batch_one_hot
                for layer in reversed(self.layers):
                    grad = layer.backprop(grad, learning_rate)
            
            # evaluate model on training set
            loss, accuracy, f1 = self.evaluate(X_train, y_train)
            print("Training loss: ", loss)
            print("Training accuracy: ", accuracy)
            print("Training f1 score: ", f1)
            print()

            # evaluate model on validation set
            loss, accuracy, f1 = self.evaluate(X_val, y_val)
            print("Validation loss: ", loss)
            print("Validation accuracy: ", accuracy)
            print("Validation f1 score: ", f1)
            print()
            print()

            # save history
            train_loss_history.append(loss)
            train_acc_history.append(accuracy)
            train_f1_history.append(f1)
            val_loss_history.append(loss)
            val_acc_history.append(accuracy)
            val_f1_history.append(f1)

        
        # plot history
        epochs_range = [i for i in range(epochs)]

        # # plot epoch vs loss
        plt.plot(epochs_range, train_loss_history, label='train')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # # plot epoch vs accuracy
        plt.plot(epochs_range, train_acc_history, label='train')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        # # plot epoch vs f1 score
        plt.plot(epochs_range, train_f1_history, label='train')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.show()

        # # plot epoch vs loss
        plt.plot(epochs_range, val_loss_history, label='val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # # plot epoch vs accuracy
        plt.plot(epochs_range, val_acc_history, label='val')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        # # plot epoch vs f1 score
        plt.plot(epochs_range, val_f1_history, label='val')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.show()

    def destroy(self):
        for layer in self.layers:
            layer.destroy()

def load_data(image_folder, limit=None):
    images = []
    image_filenames = []
    for filename in os.listdir(image_folder):
        img = cv2.imread(os.path.join(image_folder, filename))
        if img is not None:
            # convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # resize to 28x28
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
            # convert to float32
            img = img.astype(np.float32)
            images.append(img)
            image_filenames.append(filename)
            if limit is not None and len(images) == limit:
                break

    return images, image_filenames

def preprocess_data(images):
    # detection friendly preprocessing
    for i in range(len(images)):
        images[i] = cv2.dilate(images[i], (3, 3))
        # images[i] = cv2.threshold(images[i], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # images[i] = cv2.GaussianBlur(images[i], (3, 3), 0)
        # images[i] = cv2.threshold(images[i], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # convert to numpy array
    images = np.array(images)/255

    # reshape images to 28x28x1
    images = images.reshape(images.shape[0], 28, 28, 1)
    # normalize images with std and mean
    images = (images - np.mean(images)) / np.std(images)

    return images

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict labels for test images')
    parser.add_argument('--images', type=str, required=True, help='Path to the folder containing test images')
    args = parser.parse_args()

    # Load the model using pickle
    model = load_model('1705043_model.pickle')
    test_images, test_images_filenames = load_data(args.images, limit=200)
    test_images = preprocess_data(test_images)

    # predict the labels
    predicted_labels = model.predict(test_images)

    # save the results to a csv file
    results = pd.DataFrame({'FileName': test_images_filenames, 'Digit': predicted_labels})
    results.to_csv('1705043_prediction.csv', index=False)

