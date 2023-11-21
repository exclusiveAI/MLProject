import numpy as np
from exclusiveAI.components.ActivationFunctions import ELU, ReLU, Linear, Sigmoid, Tanh


def test_activation_functions():
    elu = ELU()
    relu = ReLU()
    linear = Linear()
    sigmoid = Sigmoid()
    tanh = Tanh()

    w = np.random.random(1000)

    # Relu function assert
    assert relu.function(w) == np.maximum(w, 0)
    # Relu derivative assert
    assert relu.derivative(w) == np.where(w > 0, 1, 0)
    # ELU function assert
    assert elu.function(w) == np.where(w > 0, w, np.exp(w) - 1)
    # ELU derivative assert
    assert elu.derivative(w) == np.where(w > 0, 1, np.exp(w))

    # Linear function
    assert linear.function(w) == w
    # Linear derivative
    assert linear.derivative(w) == 1

    # Sigmoid function
    assert sigmoid.function(w) == 1 / (1 + np.exp(-w))
    # Sigmoid derivative
    assert sigmoid.derivative(w) == 1 / (1 + np.exp(-w)) * (1 - 1 / (1 + np.exp(-w)))

    # Tanh function
    assert tanh.function(w) == (np.exp(w) - np.exp(-w)) / (np.exp(w) + np.exp(-w))
    # Tanh derivative
    assert tanh.derivative(w) == 1 - (np.exp(w) - np.exp(-w)) / (np.exp(w) + np.exp(-w)) ** 2


test_activation_functions()

