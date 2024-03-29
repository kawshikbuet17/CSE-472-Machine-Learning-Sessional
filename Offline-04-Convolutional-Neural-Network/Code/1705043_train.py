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
            training_loss, training_accuracy, training_f1 = self.evaluate(X_train, y_train)
            print("Training loss: ", training_loss)
            print("Training accuracy: ", training_accuracy)
            print("Training f1 score: ", training_f1)
            print()

            # evaluate model on validation set
            val_loss, val_accuracy, val_f1 = self.evaluate(X_val, y_val)
            print("Validation loss: ", val_loss)
            print("Validation accuracy: ", val_accuracy)
            print("Validation f1 score: ", val_f1)
            print()
            print()

            # save history
            train_loss_history.append(training_loss)
            train_acc_history.append(training_accuracy)
            train_f1_history.append(training_f1)
            val_loss_history.append(val_loss)
            val_acc_history.append(val_accuracy)
            val_f1_history.append(val_f1)

        # print history
        print("train loss history: ", train_loss_history)
        print("train accuracy history: ", train_acc_history)
        print("train f1 score history: ", train_f1_history)
        print("val loss history: ", val_loss_history)   
        print("val accuracy history: ", val_acc_history)
        print("val f1 score history: ", val_f1_history)

        # plot history
        epochs_range = [i for i in range(epochs)]
        print(epochs_range)

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

def save_model(model, model_name):
    # Save the model using pickle
    import pickle
    with open(model_name, 'wb') as f:
        pickle.dump(model, f)

def performance_metrics(model):
    # load test images
    test_images, test_labels = load_data('training-d', 'training-d.csv', limit = 50)
    test_images, test_labels = preprocess_data(test_images, test_labels)

    # performance metrics and confusion matrix
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score

    # predict   
    y_pred = model.predict(test_images)

    # print performance metrics
    print("Test Accuracy: ", accuracy_score(test_labels, y_pred))
    print()
    print("Confusion Matrix: \n", confusion_matrix(test_labels, y_pred))
    print()

    # print classification report
    print("Classification Report: \n", classification_report(test_labels, y_pred))

    # plot confusion matrix using seaborn
    import seaborn as sns
    import matplotlib.pyplot as plt

    cm = confusion_matrix(test_labels, y_pred)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def first_train(model_name):
    print("Training model: ", model_name)
    # load images
    train_images, train_labels = load_data('training-a', 'training-a.csv', limit = 4000)
    train_images, train_labels = preprocess_data(train_images, train_labels)

    # print shapes
    print("train_images.shape = ", train_images.shape)
    print("train_labels.shape = ", train_labels.shape)
    print("train_images[0].shape = ", train_images[0].shape)

    # suffle data
    s = np.arange(train_images.shape[0])
    np.random.shuffle(s)
    train_images = train_images[s]
    train_labels = train_labels[s]

    # split data into train and validation
    train_ratio = 0.8
    X_train = train_images[:int(train_ratio*len(train_images))]
    y_train = train_labels[:int(train_ratio*len(train_labels))]
    X_val = train_images[int(train_ratio*len(train_images)):]
    y_val = train_labels[int(train_ratio*len(train_labels)):]

    # use lenet-5 model
    model = Model(10)
    model.add(ConvolutionLayer(6, 5, 1, 1))
    model.add(ReLULayer())
    model.add(MaxPoolingLayer(pool_size=2, stride=2))
    model.add(ConvolutionLayer(16, 5, 1, 1))
    model.add(ReLULayer())
    model.add(MaxPoolingLayer(pool_size=2, stride=2))
    model.add(FlattenLayer())
    model.add(FullyConnectedLayer(output_size=120))
    model.add(ReLULayer())
    model.add(FullyConnectedLayer(output_size=84))
    model.add(ReLULayer())
    model.add(FullyConnectedLayer(output_size=10))
    model.add(SoftmaxLayer())

    # train
    model.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, learning_rate=0.00001, epochs=10, batch_size=16)
    model.destroy()

    save_model(model, '1705043_model1.pickle')
    performance_metrics(model)

def second_train(model_name):
    print("Training model: ", model_name)
    # load images
    train_images, train_labels = load_data('training-a', 'training-a.csv', limit = 4000)
    train_images, train_labels = preprocess_data(train_images, train_labels)

    # print shapes
    print("train_images.shape = ", train_images.shape)
    print("train_labels.shape = ", train_labels.shape)
    print("train_images[0].shape = ", train_images[0].shape)

    # suffle data
    s = np.arange(train_images.shape[0])
    np.random.shuffle(s)
    train_images = train_images[s]
    train_labels = train_labels[s]

    # split data into train and validation
    train_ratio = 0.8
    X_train = train_images[:int(train_ratio*len(train_images))]
    y_train = train_labels[:int(train_ratio*len(train_labels))]
    X_val = train_images[int(train_ratio*len(train_images)):]
    y_val = train_labels[int(train_ratio*len(train_labels)):]

    # use lenet-5 model
    model = Model(10)
    model.add(ConvolutionLayer(6, 5, 1, 1))
    model.add(ReLULayer())
    model.add(MaxPoolingLayer(pool_size=2, stride=2))
    model.add(ConvolutionLayer(16, 5, 1, 1))
    model.add(ReLULayer())
    model.add(MaxPoolingLayer(pool_size=2, stride=2))
    model.add(FlattenLayer())
    model.add(FullyConnectedLayer(output_size=120))
    model.add(ReLULayer())
    model.add(FullyConnectedLayer(output_size=84))
    model.add(ReLULayer())
    model.add(FullyConnectedLayer(output_size=10))
    model.add(SoftmaxLayer())

    # train
    model.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, learning_rate=0.00002, epochs=10, batch_size=16)
    model.destroy()

    save_model(model, '1705043_model2.pickle')
    performance_metrics(model)

def third_train(model_name):
    print("Training model: ", model_name)
    # load images
    train_images, train_labels = load_data('training-a', 'training-a.csv', limit = 4000)
    train_images, train_labels = preprocess_data(train_images, train_labels)

    # print shapes
    print("train_images.shape = ", train_images.shape)
    print("train_labels.shape = ", train_labels.shape)
    print("train_images[0].shape = ", train_images[0].shape)

    # suffle data
    s = np.arange(train_images.shape[0])
    np.random.shuffle(s)
    train_images = train_images[s]
    train_labels = train_labels[s]

    # split data into train and validation
    train_ratio = 0.8
    X_train = train_images[:int(train_ratio*len(train_images))]
    y_train = train_labels[:int(train_ratio*len(train_labels))]
    X_val = train_images[int(train_ratio*len(train_images)):]
    y_val = train_labels[int(train_ratio*len(train_labels)):]

    # use lenet-5 model
    model = Model(10)
    model.add(ConvolutionLayer(6, 5, 1, 1))
    model.add(ReLULayer())
    model.add(MaxPoolingLayer(pool_size=2, stride=2))
    model.add(ConvolutionLayer(16, 5, 1, 1))
    model.add(ReLULayer())
    model.add(MaxPoolingLayer(pool_size=2, stride=2))
    model.add(FlattenLayer())
    model.add(FullyConnectedLayer(output_size=120))
    model.add(ReLULayer())
    model.add(FullyConnectedLayer(output_size=84))
    model.add(ReLULayer())
    model.add(FullyConnectedLayer(output_size=10))
    model.add(SoftmaxLayer())

    # train
    model.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, learning_rate=0.00003, epochs=10, batch_size=16)
    model.destroy()

    save_model(model, '1705043_model3.pickle')
    performance_metrics(model)

def semi_final_train(model_name):
    print("Training model: ", model_name)
    # load images
    train_images_a, train_labels_a = load_data('training-a', 'training-a.csv')
    train_images_a, train_labels_a = preprocess_data(train_images_a, train_labels_a)

    train_images_b, train_labels_b = load_data('training-b', 'training-b.csv')
    train_images_b, train_labels_b = preprocess_data(train_images_b, train_labels_b)

    train_images_c, train_labels_c = load_data('training-c', 'training-c.csv')
    train_images_c, train_labels_c = preprocess_data(train_images_c, train_labels_c)

    train_images = np.concatenate((train_images_a, train_images_b, train_images_c), axis=0)
    train_labels = np.concatenate((train_labels_a, train_labels_b, train_labels_c), axis=0)

    # print shapes
    print("train_images.shape = ", train_images.shape)
    print("train_labels.shape = ", train_labels.shape)
    print("train_images[0].shape = ", train_images[0].shape)

    # suffle data
    s = np.arange(train_images.shape[0])
    np.random.shuffle(s)
    train_images = train_images[s]
    train_labels = train_labels[s]

    train_limit = 4000
    train_images = train_images[:train_limit]
    train_labels = train_labels[:train_limit]

    # split data into train and validation
    train_ratio = 0.8
    X_train = train_images[:int(train_ratio*len(train_images))]
    y_train = train_labels[:int(train_ratio*len(train_labels))]
    X_val = train_images[int(train_ratio*len(train_images)):]
    y_val = train_labels[int(train_ratio*len(train_labels)):]

    # use lenet-5 model
    model = Model(10)
    model.add(ConvolutionLayer(6, 5, 1, 1))
    model.add(ReLULayer())
    model.add(MaxPoolingLayer(pool_size=2, stride=2))
    model.add(ConvolutionLayer(16, 5, 1, 1))
    model.add(ReLULayer())
    model.add(MaxPoolingLayer(pool_size=2, stride=2))
    model.add(FlattenLayer())
    model.add(FullyConnectedLayer(output_size=120))
    model.add(ReLULayer())
    model.add(FullyConnectedLayer(output_size=84))
    model.add(ReLULayer())
    model.add(FullyConnectedLayer(output_size=10))
    model.add(SoftmaxLayer())

    # train
    model.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, learning_rate=0.00001, epochs=10, batch_size=16)
    model.destroy()

    save_model(model, '1705043_model.pickle')
    performance_metrics(model)

def final_train(model_name):
    print("Training model: ", model_name)
    # load images
    train_images_a, train_labels_a = load_data('training-a', 'training-a.csv')
    train_images_a, train_labels_a = preprocess_data(train_images_a, train_labels_a)

    train_images_b, train_labels_b = load_data('training-b', 'training-b.csv')
    train_images_b, train_labels_b = preprocess_data(train_images_b, train_labels_b)

    train_images_c, train_labels_c = load_data('training-c', 'training-c.csv')
    train_images_c, train_labels_c = preprocess_data(train_images_c, train_labels_c)

    train_images = np.concatenate((train_images_a, train_images_b, train_images_c), axis=0)
    train_labels = np.concatenate((train_labels_a, train_labels_b, train_labels_c), axis=0)

    # print shapes
    print("train_images.shape = ", train_images.shape)
    print("train_labels.shape = ", train_labels.shape)
    print("train_images[0].shape = ", train_images[0].shape)

    # suffle data
    s = np.arange(train_images.shape[0])
    np.random.shuffle(s)
    train_images = train_images[s]
    train_labels = train_labels[s]

    # split data into train and validation
    train_ratio = 0.8
    X_train = train_images[:int(train_ratio*len(train_images))]
    y_train = train_labels[:int(train_ratio*len(train_labels))]
    X_val = train_images[int(train_ratio*len(train_images)):]
    y_val = train_labels[int(train_ratio*len(train_labels)):]

    # use lenet-5 model
    model = Model(10)
    model.add(ConvolutionLayer(6, 5, 1, 1))
    model.add(ReLULayer())
    model.add(MaxPoolingLayer(pool_size=2, stride=2))
    model.add(ConvolutionLayer(16, 5, 1, 1))
    model.add(ReLULayer())
    model.add(MaxPoolingLayer(pool_size=2, stride=2))
    model.add(FlattenLayer())
    model.add(FullyConnectedLayer(output_size=120))
    model.add(ReLULayer())
    model.add(FullyConnectedLayer(output_size=84))
    model.add(ReLULayer())
    model.add(FullyConnectedLayer(output_size=10))
    model.add(SoftmaxLayer())

    # train
    model.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, learning_rate=0.00001, epochs=5, batch_size=16)
    model.destroy()

    save_model(model, '1705043_model.pickle')
    performance_metrics(model)

# write main function here
def main():
    first_train(model_name='[Train a selective] learning_rate=0.00001, epochs=10, batch_size=16')
    second_train(model_name='[Train a selective] learning_rate=0.00002, epochs=10, batch_size=16')
    third_train(model_name='[Train a selective] learning_rate=0.00003, epochs=10, batch_size=16')
    semi_final_train(model_name='[Train a,b,c selective] learning_rate=0.00001, epochs=10, batch_size=16')
    # final_train(model_name='[Train a,b,c full] learning_rate=0.00001, epochs=5, batch_size=16')

if __name__ == '__main__':
    np.random.seed(1)
    main()