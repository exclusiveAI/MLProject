from .Metric import Metric
import numpy as np


class BinaryAccuracy(Metric):
    """
		Binary Accuracy
	"""

    def __init__(self):
        super().__init__(name="binary_accuracy",
                         f=self.function)

    @staticmethod
    def function(y_true, y_pred):
        return np.average((np.round(np.abs(y_true)) == np.round(np.abs(y_pred))))
