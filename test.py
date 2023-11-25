from exclusiveAI.components import *
from exclusiveAI import neural_network
from exclusiveAI.components.Optimizers import *
from exclusiveAI.components.ActivationFunctions import *
from exclusiveAI.components.Layers import *
from exclusiveAI.components.Initializers import *
from exclusiveAI.components.LossFunctions import *
import numpy as np

x = np.random.rand(10000).reshape(-1, 1)
y = np.array(np.square(x))

x_train = x[:8000]
y_train = y[:8000]

x_val = x[8000:9000]
y_val = y[8000:9000]

x_test = x[9000:]
y_test = y[9000:]

initializer = Gaussian()

loss_func = MeanEuclideanDistance()
layers = []
input_layer = InputLayer(input_shape=1, input=x)
layer_1 = Layer(units=1, activation_func=Sigmoid(), initializer=initializer)
layer_2 = Layer(units=2, activation_func=Sigmoid(), initializer=initializer)
layer_3 = Layer(units=3, activation_func=Tanh(), initializer=initializer)
output_layer = OutputLayer(units=1, activation_function=Linear(), initializer=initializer, loss_function=loss_func)
layers.append(input_layer)
layers.append(layer_1)
layers.append(layer_2)
layers.append(layer_3)
layers.append(output_layer)

learning_rate = 0.0001
optimizer = SGD(learning_rate=learning_rate, regularization=0.0001, momentum=0.01)
nn = neural_network.neural_network(learning_rate=learning_rate, optimizer=optimizer, callbacks=[], verbose=True,
                                   layers=layers, loss=loss_func, metrics=['mse'])

nn.train(x_train, y_train, epochs=2000, batch_size=128, val=x_val, val_labels=y_val)
res = nn.predict(x_test)

print(res)
