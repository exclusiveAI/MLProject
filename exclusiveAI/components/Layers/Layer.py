__all__ = ['Layer']

from exclusiveAI.components.ActivationFunctions import ActivationFunction
from exclusiveAI.components.Initializers import Initializer
import numpy as np


class Layer:
    def __init__(self, units: int, initializer: Initializer, activation_func: ActivationFunction,
                 is_trainable: bool = True) -> object:
        # Layers
        self.next: Layer = None
        self.prev: Layer = None

        self.output = None
        self.weights = None
        self.units = units
        self.is_trainable = is_trainable
        self.activation_func = activation_func
        self.initializer = initializer
        self.is_initialized = False
        self.error = None
        self.nets = None

    def initialize(self, prev):
        self.weights = self.initializer.initialize(shape=(prev.units + 1, self.units))
        prev.next = self
        self.prev = prev
        self.is_initialized = True
        return self

    def feedforward(self, input):
        if not self.is_initialized:
            raise Exception("Layer not initialized")

        input = np.insert(input, 0, 1, axis=-1)  # adding bias to input

        self.nets = self.weights @ input
        self.output = self.activation_func.function(self.nets)
        return self.output

    def backpropagate(self):
        if not self.is_initialized:
            raise Exception("Layer not initialized")

        # calculate the product between the error signal and incoming weights from current unit
        self.error = self.next.error @ self.next.weights[1:, :].T
        self.error = self.activation_func.derivative(self.nets) * self.error
        return np.dot(self.prev.output.T, self.error)
