import tensorflow as tf
from keras.optimizers import SGD
import numpy as np
import pandas as pd

file_path = "Notebooks/MLCup/Data/training_data_split.json"
# Load training and test data from the JSON file
with open(file_path, 'r') as jsonfile:
    data_dict = json.load(jsonfile)

training_data = np.array(data_dict['training_data'])
training_labels = np.array(data_dict['training_labels'])
test_data = np.array(data_dict['test_data'])
test_labels = np.array(data_dict['test_labels'])
train_idx = np.array(data_dict['train_idx'])
test_idx = np.array(data_dict['test_idx'])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=100, activation='tanh'),
    tf.keras.layers.Dense(units=100, activation='tanh'),
    tf.keras.layers.Dense(units=100, activation='tanh'),
    tf.keras.layers.Dense(units=2, activation='linear')
])

model.compile(optimizer=SGD(learning_rate=0.005, nesterov=False, momentum=0.1, weight_decay=1e-8), loss='mean_squared_error', metrics=['mean_absolute_error'])
hist = model.fit(training_data, training_labels, workers=2, epochs=1000, batch_size=200)
res = model.evaluate(test_data, test_labels)
# model.save('model.h5')
prediction = model.predict(test_data)
print(res, hist)