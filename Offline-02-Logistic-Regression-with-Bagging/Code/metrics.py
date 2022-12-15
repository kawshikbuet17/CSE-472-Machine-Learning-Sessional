"""
Refer to: https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
"""

import numpy as np
def accuracy(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """

    # compute true positives, false positives, false negatives and true negatives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    # calculate accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return accuracy


def precision_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """

    # calculate true positives, false positives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    # calculate precision
    precision = tp / (tp + fp)

    return precision


def recall_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """

    # calculate true positives, false negatives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # calculate recall
    recall = tp / (tp + fn)

    return recall


def f1_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """

    # calculate precision and recall
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    # calculate f1 score
    f1 = 2 * precision * recall / (precision + recall)

    return f1
