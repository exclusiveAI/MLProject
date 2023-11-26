from exclusiveAI.ConfiguratorGen import ConfiguratorGen
from exclusiveAI.components.Validation import *
from exclusiveAI.datasets.monk import read_monk1
# from exclusiveAI.components import *
# from exclusiveAI import neural_network
# from exclusiveAI.components.Optimizers import *
# from exclusiveAI.components.ActivationFunctions import *
# from exclusiveAI.components.Layers import *
# from exclusiveAI.components.Initializers import *
# from exclusiveAI.components.LossFunctions import *
# from exclusiveAI.components.Callbacks import *
import numpy as np
#
#
# x1 = np.random.rand(100)
# x2 = np.random.rand(100)
# x3 = np.random.rand(100)
# #
# x = np.array([x1, x2, x3]).T
# y = np.array(np.square(x[:, :1]))
# x_train = x[:80]
# y_train = y[:80]
#
# x_val = x[80:90]
# y_val = y[80:90]
#
# x_test = x[90:]
# y_test = y[90:]
# initializer = Gaussian()
#
# wandb = wandb_Logger(run_name='test', project='exclusiveAI', config=None)
# early_stop = EarlyStopping(patience_limit=10)
#
# loss_func = MeanEuclideanDistance()
# layers = []
# input_layer = InputLayer(input_shape=3, input=x)
# layer_1 = Layer(units=1, activation_func=Sigmoid(), initializer=initializer)
# output_layer = OutputLayer(units=1, activation_function=Linear(), initializer=initializer, loss_function=loss_func)
# layers.append(input_layer)
# layers.append(layer_1)
# layers.append(output_layer)
#
# learning_rate = 0.00001
# optimizer = SGD(learning_rate=learning_rate, regularization=0.0001, momentum=0.001)
# nn = neural_network.neural_network(learning_rate=learning_rate, optimizer=optimizer, callbacks=[wandb, early_stop],
#                                    layers=layers, loss=loss_func, metrics=['mse', 'mae', 'mee'])
#
# nn.train(x_train, y_train, epochs=10000, batch_size=256, val=x_val, val_labels=y_val)
# # res = nn.predict(x_test)
# #
# # print(res)

train, test = read_monk1()
train_label = np.array(train.pop('class')).reshape(-1, 1)
test_label = np.array(test.pop('class'))
# remove_id
train = np.array(train)

values= [0.01, 0.001, 0.0001]

myconfigurator = ConfiguratorGen(random=True, num_of_configurations=10, regularizations=values, learning_rates=values,
                                    loss_functions=['mse'], optimizers=['sgd'],
                                    activation_functions=['sigmoid', 'tanh', 'relu'],
                                    number_of_units=[1, 2, 4, 8, 16], number_of_layers=[1, 2, 3, 4, 5],
                                    momentums=[0.9], initializers=['gaussian', 'uniform'], input_shapes=train.shape[-1], verbose=True
                                    )

myval = HoldOut(models=myconfigurator, input=train, target=train_label)
myval.hold_out()