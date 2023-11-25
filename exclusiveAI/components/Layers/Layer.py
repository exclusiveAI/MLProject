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
        self.name = ''
        self.output = None
        self.weights = None
        self.units = units
        self.is_trainable = is_trainable
        self.activation_func = activation_func
        self.initializer = initializer
        self.is_initialized = False
        self.error = None
        self.nets = None
        self.verbose = None

    def initialize(self, prev, name: str = '', verbose: bool = False):
        self.weights = self.initializer.initialize(shape=(prev.units + 1, self.units))
        self.name = name
        prev.next = self
        self.prev = prev
        self.verbose = verbose
        self.is_initialized = True
        if self.verbose:
            print(f"Initializing {self.name}")
            print(f"Input shape: {self.prev.units}")
            print(f"Weights shape: {self.weights.shape}")
            print(f"Output shape: {self.units}")
            print(f"Activation function: {self.activation_func.name}")
            print(f"Initializer: {self.initializer.name}")
            print(f"Trainable: {self.is_trainable}")
        return self

    def feedforward(self, input):
        if not self.is_initialized:
            raise Exception("Layer not initialized")

        local_input = np.insert(input, 0, 1, axis=-1)  # adding bias to input

        self.nets = (local_input @ self.weights)  # calculate the net input for current unit
        self.output = self.activation_func.function(self.nets)
        return self.output

    def backpropagate(self, **kwargs):
        if not self.is_initialized:
            raise Exception("Layer not initialized")

        # calculate the product between the error signal and incoming weights from current unit
        self.error = self.next.error @ self.next.weights[1:, :].T
        self.error = self.activation_func.derivative(self.nets) * self.error
        return np.dot(np.insert(self.prev.output, 0, 1, axis=-1).T, self.error)
