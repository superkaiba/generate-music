import mido as md
import numpy as np
import os
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
import random
from utils import create_model

possible_labels = np.load("possible_labels.npy")
encoded_labels = np.load("encoded_labels.npy")
data = np.load("data.npy")
model = create_model(len(possible_labels))

filepath = "weights-epoch-{epoch:02d}-loss-{loss:.4f}-accuracy-{accuracy:.4f}.hdf5" 
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath, monitor='loss', 
    verbose=0,        
    save_best_only=True,        
    mode='min'
)  

csvlogger = keras.callbacks.CSVLogger("epoch-results.csv",append=True)
opt = tf.keras.optimizers.Adam(learning_rate=0.00001)

model.compile(
    optimizer=opt, 
    loss="categorical_crossentropy", 
    metrics=['accuracy'],
    )
model.summary()
model.load_weights("weights-epoch-01-loss-0.6877-accuracy-0.8289.hdf5")
model.fit(data, encoded_labels, batch_size=128, epochs=100, callbacks=[checkpoint, csvlogger])

