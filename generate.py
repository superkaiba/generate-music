import mido as md
import numpy as np
import os
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import random

class Data():

    def __init__(self, dirname):
        self.data = []
        self.labels = []
        for file in os.listdir(dirname):
            notes = self.clean_midi_file(os.path.join(dirname, file))
            
            i = 50
            while i < len(notes):
                self.data.append(notes[i-50:i])
                self.labels.append(notes[i])
                i += 1
            
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        print(self.data)
        self.max_vector = self.data.max(axis=0).max(axis=0)
        # print(self.max_vector)
        # print(self.data)
        # print(self.labels)
        self.data = self.data/self.max_vector       # normalize data
        self.labels = self.labels/self.max_vector # normalize data
        print(self.data)
        # print(self.data)
        # print("labels", self.labels)
        # print(self.data)
        # self.data = keras.utils.normalize(self.data)
        # print(self.data)


    
    def clean_midi_file(self, filename):
        midi_file = md.MidiFile(filename)
        merged_track = md.merge_tracks([midi_file.tracks[1],midi_file.tracks[2]]) # Merge both piano tracks into one track

        relevant_notes = [x for x in merged_track if (x.type == "note_on" and (x.velocity != 0 or x.time != 0))] # Filter out useless midi messages that have no volume or time value
        relevant_notes = [np.array([x.note, x.time]) for x in relevant_notes] # Remove all information from midi messages except time since last note and pitch
        
        return relevant_notes

myData = Data("beeth")

new_midi_file = md.MidiFile()
track = new_midi_file.add_track()


model = keras.Sequential()
model.add(layers.LSTM(1024, activation='relu', recurrent_activation='relu', input_shape = (50,2), return_sequences=True))
model.add(layers.LSTM(512, activation='relu', recurrent_activation='relu', input_shape=(50, 1), return_sequences=True))
model.add(layers.LSTM(256, activation='relu', recurrent_activation='relu', input_shape=(50,1), return_sequences=False))
model.add(layers.Dense(2, activation='relu'))
model.load_weights("weights-improvement-08-0.0104-bigger.hdf5")
print(myData.data.shape)
print(len(myData.data))
random_index = random.randrange(0,len(myData.data))
sequence = myData.data[random_index].reshape(1, 50, 2)
for i in range(100):
    new_note = model.predict(sequence)
    sequence = np.append(sequence[:,1:], new_note).reshape(1,50,2)
    print(sequence)
    print(new_note)
    note_to_add = np.round(new_note * myData.max_vector)
    print(note_to_add)
    track.append(md.Message('note_on', note=int(note_to_add[0,0]), time=int(note_to_add[0,1])))

new_midi_file.save("new_midi_file.midi")



