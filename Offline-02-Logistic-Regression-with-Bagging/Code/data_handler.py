import pandas as pd
import numpy as np

def load_dataset():
    # read data from csv using pandas
    df = pd.read_csv('data_banknote_authentication.csv')
    # convert to numpy array
    data = df.values
    # get X and y
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def split_dataset(X, y, test_size = 0.10, shuffle = False):
    # split the dataset into train and test
    n = X.shape[0]
    n_test = int(n * test_size)
    n_train = n - n_test
    if shuffle:
        idx = np.random.permutation(n)
        X = X[idx]
        y = y[idx]
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    return X_train, y_train, X_test, y_test


def bagging_sampler(X, y):
    # sample with replacement
    n = X.shape[0]
    idx = np.random.choice(n, size=n, replace=True)
    X_sample = X[idx]
    y_sample = y[idx]
    return X_sample, y_sample
