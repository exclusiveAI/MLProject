import tensorflow as tf
from exclusiveAI.datasets.monk import read_monk1
from keras.optimizers import SGD
import numpy as np
from exclusiveAI.utils import confusion_matrix


train, test = read_monk1()
train_label = np.array(train.pop('class')).reshape(-1, 1)
test_label = np.array(test.pop('class')).reshape(-1, 1)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=10, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(optimizer=SGD(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train, train_label, epochs=1000)
model.evaluate(test, test_label)
model.save('model.h5')
prediction = model.predict(test)
prediction = np.round(prediction)
confusion_matrix(prediction, test_label)