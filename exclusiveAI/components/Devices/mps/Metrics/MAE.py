from .Metric import Metric
import mlx.core as mps


__all__ = ['MAE']


class MAE(Metric):
    """
    Mean Absolute Error (MAE)
    """

    def __init__(self):
        f = lambda y_pred, y_true: mps.mean(mps.abs(y_true - y_pred))
        super().__init__(name='mae', f=f)
