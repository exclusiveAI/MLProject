import numpy as np
from .LossFunction import LossFunction


class MeanEuclideanDistance(LossFunction):
    def __init__(self):
        super().__init__()
        self.name = "Mean Euclidean Distance"
        self.loss_function = lambda y_true, y_pred: np.mean(np.linalg.norm(y_true - y_pred, axis=1))
        self.loss_function_derivative = lambda y_true, y_pred: \
            (y_true - y_pred) / np.linalg.norm(y_true - y_pred, axis=1).reshape(-1, 1)
