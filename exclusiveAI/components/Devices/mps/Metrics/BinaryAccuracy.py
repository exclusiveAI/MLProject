from .Metric import Metric
import mlx.core as mps

class BinaryAccuracy(Metric):
	"""
	Binary Accuracy
	"""
	def __init__(self):
		name = "binary_accuracy"
		f = lambda y_true, y_pred: mps.sum(mps.abs(y_true, dtype='int8') == mps.abs(y_pred, dtype='int8')) / y_true.shape[0]
		super().__init__(name, f=f)