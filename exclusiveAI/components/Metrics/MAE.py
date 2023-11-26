from .Metric import Metric
import numpy as np

__all__ = ['MAE']
class MAE(Metric):
    def __init__(self):
        f = lambda y_pred, y_true: np.mean(np.abs(y_pred - y_true))
        super().__init__(name = 'mae', f=f)

        
        