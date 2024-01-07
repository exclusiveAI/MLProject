from .Metric import Metric
import numpy as np

__all__ = ['MEE']


class MEE(Metric):
    """
    Mean Euclidean Error (MEE)
    """
    def __init__(self):
        super().__init__(name='mee',
                         f=self.function
                         )

    @staticmethod
    def function(y_true, y_pred):
        return np.mean((np.linalg.norm(y_true - y_pred, axis=1)), dtype='float32')
