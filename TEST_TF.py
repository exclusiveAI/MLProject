import tensorflow as tf
from exclusiveAI.datasets.monk import read_monk1
from keras.optimizers import SGD
import numpy as np
from exclusiveAI.utils import confusion_matrix, one_hot_encoding

train, train_label, test, test_label = read_monk1()

train = np.array(train.values.tolist())
train_label = train_label.reshape(-1, 1)
train = one_hot_encoding(train)
test = np.array(test.values.tolist())
test_label = test_label.reshape(-1, 1)
test = one_hot_encoding(test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=2, activation='sigmoid'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(optimizer=SGD(), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train, train_label, epochs=1000)
model.evaluate(test, test_label)
model.save('model.h5')
prediction = model.predict(test)
prediction = np.round(prediction)
confusion_matrix(prediction, test_label)