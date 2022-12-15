from data_handler import bagging_sampler
import numpy as np

class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator):
        """
        :param base_estimator:
        :param n_estimator:
        :return:
        """
        self.base_estimator = base_estimator
        self.n_estimator = n_estimator
        self.estimators = []

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2

        for i in range(self.n_estimator):
            X_, y_ = bagging_sampler(X, y)
            self.estimators.append(self.base_estimator.fit(X_, y_))
        return self

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        apply majority voting
        :param X:
        :return:
        """
        y_pred = np.zeros(X.shape[0])
        for estimator in self.estimators:
            y_pred += estimator.predict(X)
        y_pred = y_pred / self.n_estimator
        y_pred = np.array(y_pred >= 0.5, dtype='int')
        return y_pred