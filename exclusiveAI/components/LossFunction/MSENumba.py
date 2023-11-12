import numpy as np
from numba import njit, prange
from numpy import ndarray

from .LossFunction import LossFunction


class MSENumba(LossFunction):
    def __init__(self):
        self.__init__()
        self.__name__ = "MSENumba"
        self.loss_function = lambda y_true, y_pred: self.mse_numba_loss(y_true=y_true, y_pred=y_pred)
        self.gradient = lambda y_true, y_pred: self.mse_numba_derivative(y_true=y_true, y_pred=y_pred)

    @njit(cache=True, fastmath=True)
    def mse_numba_loss(self, y_true, y_pred) -> np.ndarray:
        return np.mean(np.sum(np.square(y_true - y_pred), axis=1))

    @njit(cache=True, fastmath=True)
    def mse_numba_derivative(self, y_true, y_pred) -> np.ndarray:
        return np.mean(np.subtract(y_true, y_pred))
