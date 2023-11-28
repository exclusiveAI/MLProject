from exclusiveAI.components.Layers.Layer import Layer
from exclusiveAI.components.ActivationFunctions import Linear
from exclusiveAI.components.Initializers import Initializer
import numpy as np


class InputLayer(Layer):
    def __init__(self, input_len, units: int):
        super().__init__(
            units=units,
            initializer=Initializer(),
            is_trainable=False,
            activation_func=Linear(),
        )
        self.input_len = input_len
        self.input = None

    def __str__(self) -> str:
        return super().__str__()

    def initialize(self, name: str = '', verbose: bool = False, **kwargs):
        self.weights = self.initializer.ones(shape=(self.units+1, self.units))
        self.name = name
        self.verbose = verbose
        self.is_initialized = True
        if self.verbose:
            print(f"Initializing {self.name}")
            print(f"Input shape: {self.input_len}")
            print(f"Input: {self.input}")
            print(f"Output shape: {self.units}")
            print(f"Activation function: {self.activation_func.name}")
            print(f"Initializer: {self.initializer.name}")
            print(f"Trainable: {self.is_trainable}")
        return self

    def feedforward(self, input):
        self.input = input  # saving input for weights update during backpropagation
        return super().feedforward(input)

    def backpropagate(self, **kwargs):
        if not self.is_initialized:
            raise Exception("Layer not initialized")

        # calculate the product between the error signal and incoming weights from current unit
        self.error = self.next.error @ self.next.weights[1:, :].T
        self.error = self.activation_func.derivative(self.nets) * self.error
        return np.dot(np.insert(self.input, 0, 1, axis=-1).T, self.error)
