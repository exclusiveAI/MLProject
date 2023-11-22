from exclusiveAI.components.Layers.Layer import Layer
from exclusiveAI.components.ActivationFunctions import Linear
import numpy as np


class InputLayer(Layer):
    def __init__(self, input_shape, input):
        super().__init__(
            is_trainable=False,
            activation_func=Linear(),
        )

        self.input_shape = input_shape,
        self.input = input

    def initialize(self, **kwargs):
        self.weights = self.initializer.ones(shape=(self.input_shape[-1] + 1, self.units))
        self.is_initialized = True
        return self

    def feedforward(self, input):
        self.input = input  # saving input for weights update during backpropagation
        return super().feedforward(input)

    def backpropagate(self):
        if not self.is_initialized:
            raise Exception("Layer not initialized")

        # calculate the product between the error signal and incoming weights from current unit
        self.error = self.next.error @ self.next.weights[1:, :].T
        self.error = self.activation_func.derivative(self.nets) * self.error
        return np.dot(self.input.T, self.error)
