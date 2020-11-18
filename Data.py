import mido as md
import numpy as np
import sys
import os
from params import NUM_TIMESTEPS , MAX_VECTOR, MIN_VECTOR, POSSIBLE_TIMES
from utils import clean_midi_file, normalize, denormalize

class Data():

    def __init__(self, dirname):
        self.data = []
        self.labels = []

        for file in os.listdir(dirname):
            notes = clean_midi_file(os.path.join(dirname, file))
            i = NUM_TIMESTEPS
            while i < len(notes):
                self.data.append(notes[i-NUM_TIMESTEPS:i])
                self.labels.append(notes[i])
                i += 1
            
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        self.data = normalize(self.data)

        self.possible_labels = self.get_possible_labels()
        self.possible_labels_dict = {}

        for index, possible_label in enumerate(self.possible_labels):
            self.possible_labels_dict[repr(possible_label)] = index

        self.encoded_labels = []

        for label in self.labels:
            self.encoded_labels.append(self.one_hot_encode(repr(label)))

        self.encoded_labels = np.array(self.encoded_labels)

    def get_possible_labels(self):
        possible_notes = np.arange(start=21,stop=109)
        possible_times = np.array(POSSIBLE_TIMES)
        possible_velocities = np.array([0,1])
        possible_labels = np.array(np.meshgrid(possible_notes, possible_times,possible_velocities)).T.reshape(-1,3)

        return possible_labels

    def one_hot_encode(self, note):
        one_hot_array = np.zeros(len(self.possible_labels))
        one_hot_array[self.possible_labels_dict[note]] = 1
        return one_hot_array

myData = Data("test-data")

np.save("test-data.npy",myData.data)
# np.save("possible_labels.npy",myData.possible_labels)
# np.save("encoded_labels.npy",myData.encoded_labels)