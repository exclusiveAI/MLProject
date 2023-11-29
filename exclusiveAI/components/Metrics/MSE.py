from .Metric import Metric
import numpy as np

__all__ = ['MSE']


class MSE(Metric):
    def __init__(self):
        f = lambda y_pred, y_true: np.mean(np.sum(np.square((y_pred - y_true), dtype='float32'), axis=1), dtype='float32')
        super().__init__(name='mse', f=f)
