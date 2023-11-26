from exclusiveAI import utils
# from exclusiveAI.components import *
# from exclusiveAI import neural_network
# from exclusiveAI.components.Optimizers import *
# from exclusiveAI.components.ActivationFunctions import *
# from exclusiveAI.components.Layers import *
# from exclusiveAI.components.Initializers import *
# from exclusiveAI.components.LossFunctions import *
# from exclusiveAI.components.CallBacks import *
import numpy as np
#
#
x1 = np.random.rand(100)
x2 = np.random.rand(100)
x3 = np.random.rand(100)
#
x = np.array([x1, x2, x3]).T
y = np.array(np.square(x[:, :1]))
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

train, y_train, tmp1, y_tmp1, train_indexes, tmp_indexes = utils.train_split(x, y, split_size=0.3, random_state=42)
val, y_val, test, y_test, val_indexes, test_indexes = utils.train_split(tmp1, y_tmp1, split_size=0.33, random_state=42)
print(len(train), len(val), len(test), tmp_indexes, val_indexes, test_indexes)
