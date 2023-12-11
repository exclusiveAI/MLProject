from .Metric import Metric
import numpy as np
import mlx.core as mps

__all__ = ['MSE']


class MSE(Metric):
    """
    Mean squared error (MSE)
    """
    def __init__(self):
        f = lambda y_pred, y_true: mps.mean(mps.sum(mps.square(y_pred - y_true), axis=1))
        super().__init__(name='mse', f=f)
