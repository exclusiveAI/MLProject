from .Metric import Metric
import numpy as np

__all__ = ['MAE']


class MAE(Metric):
    """
    Mean Absolute Error (MAE)
    """

    def __init__(self):
        f = lambda y_pred, y_true: np.mean(np.abs(y_true - y_pred), dtype='float32')
        super().__init__(name='mae', f=f)
