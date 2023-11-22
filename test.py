from exclusiveAI.components import *
from exclusiveAI import neural_network
from exclusiveAI.components.Optimizers import *
from exclusiveAI.components.ActivationFunctions import *
from exclusiveAI.components.Layers import *
from exclusiveAI.components.Initializers import *
from exclusiveAI.components.LossFunctions import *
import numpy as np

x = np.array([1, 2, 3, 4, 5])
y_true = np.array([1, 4, 9, 16, 25])

x_test = np.array([6, 7])
y_test = np.array([36, 49])


initializer = Gaussian()

loss_func = MeanEuclideanDistance()
layers = []
input_layer = InputLayer(input_shape=5, input=x)
layer_1 = Layer(units=3, activation_func=Tanh(), initializer=initializer)
output_layer = OutputLayer(units=1, activation_function=Linear(), initializer=initializer, loss_function=loss_func)
layers.append(input_layer)
layers.append(layer_1)
layers.append(output_layer)


learning_rate = 0.01
act_func = Sigmoid()
optimizer = SGD(learning_rate=learning_rate, regularization=0.0001, activation_func=act_func, momentum=0.01)
nn = neural_network.neural_network(learning_rate=learning_rate, optimizer=optimizer, callbacks=[], verbose=True,
                                   layers=layers, loss=loss_func)

nn.train(x, y_true, epochs=10)
res = nn.predict(x_test)


print(res)