__all__ = ['Layer']

from exclusiveAI.components.ActivationFunctions import ActivationFunction
from exclusiveAI.components.Initializers import Initializer
import numpy as np


class Layer:
    """
    Represents a generic layer of the neural network.

    Args:
        units (int): Number of units in the layer.
        initializer (Initializer): Initializer object to initialize the weights.
        activation_func (ActivationFunction): Activation function object to apply to the nets.
        is_trainable (bool): Whether the layer is trainable or not. Defaults to True.
    Attributes:
        next (Layer): the next Layer of the current one.
        prev (Layer): the previous Layer of the current one.
        name (str): the name of the layer.
        output (np.ndarray): the output of the layer.
        weights (np.ndarray): the connection weights of the layer.
        units (int): the number of units in the layer.
        is_trainable (bool): whether the layer is trainable or not.
        activation_func (ActivationFunction): the activation function object to apply to the nets.
        is_initialized (bool): whether the layer is initialized or not.
        error (np.ndarray): the error of the layer.
        nets (np.ndarray): the nets of the layer.
        verbose (bool): whether to print the information about the layer.
    """
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
        
    def __str__(self) -> str:
        """
        Print the name of the layer and the number of units in it.
        Returns:
            str: the name of the layer and the number of units in it.
        """
        return f"{self.name} - {self.units}"

    def initialize(self, prev, name: str = '', verbose: bool = False):
        """
        Layer initialization.

        Args:
            prev (Layer): the previous Layer of the current one.
            name (str): the name of the layer.
            verbose (bool): whether to print the information about the layer.

        Returns:
            Layer: the current Layer.

        Raises:
            Exception: if the layer is already initialized.

        Notes:
            The first unit of the layer is the bias unit.
            The weights of the layer are initialized using the `initializer` object.
            The name of the layer is set to `name` if provided, otherwise it is set to the class name.
            The previous layer of the current one is set to `prev`.
            The verbose flag is set to `verbose` if provided, otherwise it is set to False.
            The is_initialized flag is set to True.
            If the verbose flag is True, it prints the information about the layer.
            The weights of the layer are initialized using the `initializer` object.
            The output of the layer is initialized to None.
            The error of the layer is initialized to None.
            The nets of the layer are initialized to None.
        """
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
        """
        Apply the feedforward step to the input.

        Args:
            input: input to the layer.

        Raises:
            Exception: if the layer is not initialized.

        Returns:
            np.ndarray: the output of the layer (activation function applied to the nets)

        """
        if not self.is_initialized:
            raise Exception("Layer not initialized")

        local_input = np.insert(input, 0, 1, axis=-1)  # adding bias to input

        self.nets = (local_input @ self.weights)  # calculate the net input for current unit
        self.output = self.activation_func.function(self.nets)
        return self.output

    def backpropagate(self, **kwargs):
        """
        Backpropagation step.

        Raises:
            Exception: if the layer is not initialized.

        Returns:

        """
        if not self.is_initialized:
            raise Exception("Layer not initialized")

        previous_output = np.insert(self.prev.output, 0, 1, axis=-1)  # adding bias to input

        next_weights = self.next.weights[1:, :].T

        # calculate the product between the error signal and incoming weights from current unit
        self.error = np.matmul(self.next.error, next_weights, dtype=np.longdouble)
        self.error = self.error * self.activation_func.derivative(self.nets)

        res = np.dot(previous_output.T, self.error)

        return res

    def get_weights(self):
        return self.weights

    def get_error(self):
        return self.error

    def set_weights(self, weights):
        self.weights = weights
