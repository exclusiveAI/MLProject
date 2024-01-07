from .Metric import Metric
import numpy as np

__all__ = ['MAE']


class MAE(Metric):
    """
    Mean Absolute Error (MAE)
    """

    def __init__(self):
        super().__init__(name='mae', f=self.function)

    @staticmethod
    def function(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred, dtype='float32'), dtype='float32')
