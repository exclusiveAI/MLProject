from exclusiveAI.components.Layers.Layer import Layer
from exclusiveAI.components.ActivationFunctions import Linear
from exclusiveAI.components.Initializers import *
import numpy as np


class InputLayer(Layer):
    """
    Specialization of the Layer class to represent an input layer.

    Args:
        input_len (int): The length of the input.
        units (int): The number of units in the layer.
    Attributes:
        units (int): The number of units in the layer.
        initializer (Initializer): The initializer to use.
        is_trainable (bool): Whether the layer is trainable or not.
        activation_func (ActivationFunction): The activation function to use.
        input_len (int): The length of the input.
        input (np.ndarray): The input to the layer.
    """
    def __init__(self, input_len, units: int):
        super().__init__(
            units=units,
            initializer=Uniform(low=-1, high=1),
            is_trainable=False,
            activation_func=Linear(),
        )
        self.input_len = input_len
        self.input = None

    def __str__(self) -> str:
        return super().__str__()

    def initialize(self, name: str = '', verbose: bool = False, **kwargs):
        # self.weights = self.initializer.ones(shape=(self.units+1, self.units))
        self.weights = self.initializer.initialize(shape=(self.units + 1, self.units))
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
        if not self.is_initialized:
            raise Exception("Layer not initialized")
        self.input = input
        local_input = input

        #NOTE: this layer simply uses the local input as nets and an identity function as activation function
        self.nets = local_input # calculate the net input for current unit
        self.output = self.activation_func.function(self.nets)
        return self.input

    def backpropagate(self, **kwargs):
        if not self.is_initialized:
            raise Exception("Layer not initialized")

        input = np.insert(self.input, 0, 1, axis=-1)  # adding bias to input
        next_weights = self.next.weights[1:, :].T
        # calculate the product between the error signal and incoming weights from current unit
        self.error = self.next.error @ next_weights
        self.error = self.activation_func.derivative(self.nets) * self.error
        return np.dot(input.T, self.error)
