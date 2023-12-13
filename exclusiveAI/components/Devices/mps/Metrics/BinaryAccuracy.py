from .Metric import Metric
import mlx.core as mps

class BinaryAccuracy(Metric):
	"""
	Binary Accuracy
	"""
	def __init__(self):
		name = "binary_accuracy"
		super().__init__(name, f=self.function)

	@staticmethod
	def function(y_true, y_pred):
		return mps.sum(mps.abs(y_true, dtype='int8') == mps.abs(y_pred, dtype='int8')) / y_true.shape[0]
