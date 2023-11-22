from exclusiveAI.components.Layers.Layer import Layer
from exclusiveAI.components.ActivationFunctions import ActivationFunction


class OutputLayer(Layer):
    def __init__(self, activation_function: ActivationFunction, units: int):
        super().__init__(
            activation_func=activation_function,
            units=units,
        )

    def get_output(self):
        return self.output
