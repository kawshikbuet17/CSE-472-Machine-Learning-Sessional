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
        loss /= len(y)

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
        # plot epoch vs loss
        plt.plot(train_loss_history, label='train')
        plt.plot(val_loss_history, label='val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # plot epoch vs accuracy
        plt.plot(train_acc_history, label='train')
        plt.plot(val_acc_history, label='val')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        # plot epoch vs f1 score
        plt.plot(train_f1_history, label='train')
        plt.plot(val_f1_history, label='val')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.show()



