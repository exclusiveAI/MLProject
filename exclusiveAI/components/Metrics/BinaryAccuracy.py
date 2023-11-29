from .Metric import Metric
import numpy as np

class BinaryAccuracy(Metric):
	def __init__(self):
		name = "binary_accuracy"
		f = lambda y_true, y_pred: np.average((np.round(np.abs(y_true)) == np.round(np.abs(y_pred))))
		super().__init__(name, f=f)