import numpy as np

class LogisticRegression:
    def __init__(self, params):
        """
        figure out necessary params to take as input
        :param params:
        """
        self.params = params
        self.W = None
        self.b = None

    # sigmoid function
    def sigmoid(self, z):
        """
        :param z:
        :return:
        """
        return 1 / (1 + np.exp(-z))

    # function to initialize weights and bias
    def initialize_weights_and_bias(self, d):
        # self.W = np.zeros((d,))
        self.W = np.random.uniform(low=-1, high=1, size=(d,))
        self.b = 0

    # function for forward pass
    def forward_pass(self, X):
        z = np.dot(X, self.W) + self.b
        y_hat = self.sigmoid(z)
        return y_hat

    # function to calculate the loss
    def loss(self, X, y):
        z = np.dot(X, self.W) + self.b
        y_hat = self.sigmoid(z)
        loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return loss

    # function to compute the gradient
    def gradient(self, X, y, y_hat):
        dw = np.dot(X.T, (y_hat - y)) / y.shape[0]
        db = np.mean(y_hat - y)
        return dw, db

    # function to update the weights and bias
    def update(self, dw, db):
        self.W -= self.params['lr'] * dw
        self.b -= self.params['lr'] * db

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2

        # initialize weights and bias
        n, d = X.shape

        # initialize weights and bias
        self.initialize_weights_and_bias(d)

        # gradient descent
        for i in range(self.params['n_iters']):
            # forward pass
            y_hat = self.forward_pass(X)

            # compute loss
            loss = self.loss(X, y)

            # compute gradients
            dw, db = self.gradient(X, y, y_hat)

            # update parameters
            self.update(dw, db)

            # if i % 100 == 0:
            #     print(f'iter {i}: loss = {loss}')

        return self

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        :param X:
        :return:
        """
        z = np.dot(X, self.W) + self.b
        y_hat = self.sigmoid(z)
        y_pred = np.array(y_hat >= 0.5, dtype='int')
        return y_pred
