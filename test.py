import mido as md
import numpy as np
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
# TODO: Figure out how to make labels arrays of length 2
data = np.array([[1,50],[2,60],[3,70]])
length = len(data)
data = np.reshape(data, (1, 3, 2))
labels = np.array([[2,60],[3,70],[4,80]])
labels = np.reshape(labels, (1, 3, 2))
print(labels.shape)
print(data.shape)

model = keras.Sequential()
model.add(layers.LSTM(128, activation='relu', recurrent_activation='relu', input_shape=(None, 2)))
model.add(layers.Dense(3, activation='softmax'))
model.compile(
    optimizer="rmsprop", 
    loss="categorical_crossentropy", 
    metrics=['accuracy']
    )
model.summary()
model.fit(data, labels, batch_size=1,epochs=1)