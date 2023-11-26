from .Metric import Metric
import numpy as np

__all__ = ['MEE']


class MEE(Metric):
    def __init__(self):
        super().__init__(name='mee',
                         f=lambda y_pred, y_true: np.mean((np.linalg.norm(np.subtract(y_true, y_pred)))),
                         )
