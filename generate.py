import mido as md
import numpy as np
import os
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from utils import create_model, get_start_of_midi_file, denormalize, normalize
import random
from params import NUM_TIMESTEPS

# put path to weights folder that you want to use here
weights_path = "weights-epoch-01-loss-0.6877-accuracy-0.8289.hdf5"

# put path to test-data .npy files that you want to use here
possible_labels = np.load("possible_labels.npy")
data = np.load("test-data.npy")

model = create_model(len(possible_labels))
model.load_weights(weights_path)

new_midi_file = md.MidiFile()
track = new_midi_file.add_track()

random_index = random.randrange(0,len(data))
sequence = data[random_index].reshape(1, NUM_TIMESTEPS, 3)

for i in range(NUM_TIMESTEPS):
    new_note = denormalize(sequence[0,i]) 
    track.append(md.Message('note_on', note=int(new_note[0]), time=int(new_note[1]), velocity=int(new_note[2])))

track.append(md.Message('note_on', note=108, time=500, velocity=100))

for i in range(1500):
    new_note = possible_labels[np.argmax(model.predict(sequence), axis=-1)]

    sequence = np.append(sequence[:,1:], normalize(new_note)).reshape(1,NUM_TIMESTEPS,3)
    velocity = 0
    if new_note[0,2] != 0:
        velocity = 50
    track.append(md.Message('note_on', note=int(new_note[0,0]), time=int(new_note[0,1]), velocity = velocity))
    print(i)

# print("Random index: ", random_index)
# Put name of file to save to here
new_midi_file.save("random-index-{}-epoch-01-loss-0.6877-accuracy-0.8289.midi".format(random_index))

# good weights to generate:
# MODEL 1
# -epoch-04-loss-0.6966-accuracy-0.8201
# epoch-11-loss-0.4558-accuracy-0.8995
# MODEL 2 (500 timesteps)
# epoch-02-loss-0.7290-accuracy-0.8398.hdf5