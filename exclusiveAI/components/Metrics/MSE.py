from .Metric import Metric
import numpy as np

__all__ = ['MSE']


class MSE(Metric):
    """
    Mean squared error (MSE)
    """
    def __init__(self):
        super().__init__(name='mse', f=self.function)

    @staticmethod
    def function(y_pred, y_true):
        return np.mean(np.sum(np.square(y_pred - y_true, dtype='float32'), axis=1, dtype='float32'), dtype='float32')
