import tensorflow as tf
from keras.optimizers import SGD
import numpy as np
import json

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
    tf.keras.layers.Dense(units=20, activation='sigmoid'),
    tf.keras.layers.Dense(units=15, activation='sigmoid'),
    tf.keras.layers.Dense(units=10, activation='sigmoid'),
    tf.keras.layers.Dense(units=3, activation='linear')
])

model.compile(optimizer=SGD(learning_rate=0.05, nesterov=True, momentum=0.5, weight_decay=1e-8), loss='mean_squared_error', metrics=['mean_absolute_error'])
model.fit(training_data, training_labels, epochs=1000, batch_size=200)
model.evaluate(test_data, test_labels)
model.save('model.h5')
prediction = model.predict(test_data)
