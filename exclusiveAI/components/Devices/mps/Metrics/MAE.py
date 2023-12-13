from .Metric import Metric
import mlx.core as mps


__all__ = ['MAE']


class MAE(Metric):
    """
    Mean Absolute Error (MAE)
    """

    def __init__(self):
        super().__init__(name='mae', f=self.function)

    @staticmethod
    def function(y_true, y_pred):
        return mps.mean(mps.abs(y_true - y_pred))