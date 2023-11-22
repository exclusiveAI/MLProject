from .Metric import Metric
import numpy as np

__all__ = ['MEE']
class MSE(Metric):
    def __init__(self):
        f = lambda y_pred, y_true: np.mean((np.linalg.norm(y_pred, y_true)))
        super().__init__(name = 'MEE', f=f)

        
        