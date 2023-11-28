import numpy as np

from exclusiveAI.components.Layers.Layer import Layer
from exclusiveAI.components.ActivationFunctions import ActivationFunction
from exclusiveAI.components.Initializers import Initializer
from exclusiveAI.components.LossFunctions import LossFunction


class OutputLayer(Layer):
    def __init__(self, activation_function: ActivationFunction, units: int, initializer: Initializer,
                 loss_function: LossFunction):
        super().__init__(
            initializer=initializer,
            activation_func=activation_function,
            units=units,
        )
        self.loss_function = loss_function

    def get_output(self):
        return self.output

    def backpropagate(self, y_true: np.ndarray):
        if not self.is_initialized:
            raise Exception("Layer not initialized")
        if y_true.shape == (y_true.shape[0],):
            y_true = y_true.reshape(-1, 1)


        # calculate the product between the error signal and incoming weights from current unit
        loss_btw_output_and_y_true = self.loss_function.function_derivative(y_true, self.output)
        self.error = (self.activation_func.derivative(self.nets) * loss_btw_output_and_y_true)

        return np.dot(np.insert(self.prev.output, 0, 1, axis=-1).T, self.error)
