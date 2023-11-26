from .Metric import Metric
import numpy as np

__all__ = ['MSE']


class MSE(Metric):
    def __init__(self):
        f = lambda y_pred, y_true: np.mean((y_pred - y_true) ** 2)
        super().__init__(name='mse', f=f)
