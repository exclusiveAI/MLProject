from .Metric import Metric
import numpy as np
import mlx.core as mps

__all__ = ['MEE']


class MEE(Metric):
    """
    Mean Euclidean Error (MEE)
    """
    def __init__(self):
        super().__init__(name='mee',
                         f=self.function,
                         )

    @staticmethod
    def function(y_true, y_pred):
        return mps.mean(mps.sqrt(mps.sum(mps.square(y_pred - y_true), axis=1)))
