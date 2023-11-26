from .Metric import Metric
import numpy as np

class BinaryAccuracy(Metric):
	def __init__(self):
		name = "binary_accuracy"
		f = lambda y_true, y_pred: np.average((y_true == np.rint(y_pred)))
		super().__init__(name, f)