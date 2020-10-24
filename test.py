import mido as md
import numpy as np
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
# TODO: - Figure out how to one-hot-encode arrays
#       - Make dictionary with {number : (pitch, time)} to decode result given by neural network

encoder = LabelBinarizer()
data = np.array([[[1,50],[2,60],[3,70]], [[4,80],[5,90],[6,100]]])

data = np.reshape(data, (2, 3, 2))

labels = [[2,60],[3,70],[4,80],[5,90],[4,80],[7,110]]
possible_labels = np.unique(np.array(labels), axis=0)
print(possible_labels)
print(possible_labels[0])
number_classifier_dict = {i:possible_labels[i] for i in range(len(possible_labels))}
print(number_classifier_dict)


# model = keras.Sequential()
# model.add(layers.LSTM(128, activation='relu', recurrent_activation='relu', input_shape=(None, 2)))
# model.add(layers.Dense(3, activation='softmax'))
# model.compile(
#     optimizer="rmsprop", 
#     loss="categorical_crossentropy", 
#     metrics=['accuracy']
#     )
# model.summary()
# model.fit(data, labels, batch_size=1,epochs=1)