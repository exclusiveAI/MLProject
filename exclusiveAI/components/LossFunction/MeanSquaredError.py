import numpy as np
from numpy import ndarray

from .LossFunction import LossFunction


class MeanSquaredError(LossFunction):
    def __init__(self):
        super().__init__()
        self.name = "Mean Squared Error"

        self.loss_function = lambda y_true, y_pred: self.mse_loss(y_true, y_pred)
        self.loss_function_derivative = lambda y_true, y_pred: self.mse_loss_derivative(y_true, y_pred)

    @staticmethod
    def mse_loss(y_true, y_pred) -> ndarray:
        diff = np.subtract(y_true, y_pred)
        return np.mean(np.sum(np.square(diff), axis=1))

    @staticmethod
    def mse_loss_derivative(y_true, y_pred) -> ndarray:
        diff = np.subtract(y_true, y_pred)
        return (-2 * diff) * diff
