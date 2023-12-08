# from exclusiveAI.utils import confusion_matrix, one_hot_encoding, train_split
# from exclusiveAI.components.Initializers import *
# from exclusiveAI.datasets.monk import read_monk1
# from exclusiveAI.components.CallBacks import *
# from exclusiveAI.Composer import Composer
import numpy as np
#
# train, train_label, test, test_label = read_monk1()
#
# train = np.array(train.values.tolist())
# train = one_hot_encoding(train)
# test = np.array(test.values.tolist())
# test = one_hot_encoding(test)
#
# train, train_label, val, val_label, _, _ = train_split(train, train_label)
# # train = np.array([[1, 2, 3], [4, 5, 6]])
# # train_label = np.array([[1], [0]])
# # test = np.array([[1, 2, 3], [4, 5, 6]])
# # test_label = np.array([[1], [0]])
#
# ea = EarlyStoppingCallback(patience_limit=3)
# uniform = Uniform(low=-1, high=1)
#
# model = Composer(regularization=0.001,
#                  learning_rate=0.01,
#                  loss_function='mse',
#                  optimizer='adam',
#                  nesterov=True,
#                  activation_functions=['sigmoid', 'sigmoid'],
#                  num_of_units=[10],
#                  num_layers=1,
#                  momentum=0,
#                  initializers=uniform,
#                  input_shape=train.shape,
#                  verbose=True,
#                  callbacks=[ea],
#                  ).compose()
#
# model.train(train, train_label, val, val_label, epochs=100, name='test1')
# res = model.evaluate(input=test, input_label=test_label)
# print(res)
# # model.plot_history('binary_accuracy')

from exclusiveAI.datasets.mlcup import read_cup_training_dataset, train_val_test_split

train, labels = read_cup_training_dataset()

train_val_test_split(train, labels)

# print(train, labels, train.shape, labels.shape)
