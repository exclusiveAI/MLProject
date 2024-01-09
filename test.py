from exclusiveAI.ConfiguratorGen import ConfiguratorGen
from exclusiveAI.Composer import Composer
from exclusiveAI.components.Validation.HoldOut import parallel_hold_out, hold_out
from exclusiveAI.datasets.monk import read_monk2
from exclusiveAI.utils import one_hot_encoding, plot_history

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

train, train_label, test, test_label = read_monk2()

train = one_hot_encoding(train)
test = one_hot_encoding(test)

# ea = EarlyStoppingCallback(patience_limit=50, restore_weights=False)
values = list(np.arange(0.5, 0.9, 0.1))
values2 = [1e-7, 0]

myConfigurator = ConfiguratorGen(random=False, learning_rates=values, regularizations=values2,
                                 loss_function=['mse'], optimizer=['adam'],
                                 activation_functions=['sigmoid'],
                                 number_of_units=[2, 3, 4], number_of_layers=[1],
                                 momentums=[0.96], initializers=["uniform"],
                                 input_shapes=train.shape,
                                 verbose=False, nesterov=True,
                                 callbacks=["earlystopping"], output_activation='sigmoid'
                                 ).get_configs()
if __name__ == '__main__':
    configs = parallel_hold_out(configs=myConfigurator, training=train, training_target=train_label, num_models=3,
                                epochs=200, batch_size=32, assessment=False, number_of_initializations=1)
    configs2 = hold_out(configs=myConfigurator, training=train, training_target=train_label, num_models=3,
                        epochs=200, batch_size=32, assessment=False, number_of_initializations=1)

    configs3 = parallel_hold_out(configs=myConfigurator, training=train, training_target=train_label, num_models=3,
                                 epochs=200, batch_size=32, workers=8, assessment=False, number_of_initializations=1)

    config = configs[0]
    config2 = configs2[0]
    config3 = configs3[0]
    print("Model found:", config)
    model = Composer(config=config).compose()
    print("Model 1 train")
    model.train(train.copy(), train_label.copy(), epochs=200, batch_size=32)

    history_mse = model.history['mee']
    res = model.evaluate(input=test, input_label=test_label)

    print("Model found:", config2)
    model2 = Composer(config=config2).compose()
    print("Model 2 train")
    model2.train(train.copy(), train_label.copy(), epochs=200, batch_size=32)

    history_mse2 = model2.history['mee']
    res2 = model2.evaluate(input=test, input_label=test_label)

    print("Model found:", config3)
    model3 = Composer(config=config3).compose()
    print("Model 3 train")
    model3.train(train.copy(), train_label.copy(), epochs=200, batch_size=32)

    history_mse3 = model3.history['mee']
    plot_history(lines={'mee1': history_mse, 'mee2': history_mse2, 'mee3': history_mse3})
    res3 = model3.evaluate(input=test, input_label=test_label)

    print(res, res2, res3)
    print(configs)
    print(configs2)
    print(configs3)
    # prediction = model.predict(input=test)

    # prediction = 1 if x > 0.5 else 0
    # prediction = np.round(prediction)

    # confusion_matrix(prediction, test_label)
