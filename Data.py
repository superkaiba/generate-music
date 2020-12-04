import mido as md
import numpy as np
import sys
import os
from params import *
from utils import * 

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

        self.pitch_labels = self.labels[:,0]
        self.vel_labels = self.labels[:,2]

        self.encoded_vel_labels = np.array([self.one_hot_encode_vel(vel) for vel in self.vel_labels])

        self.duration_labels = self.labels[:, 1]
        self.duration_labels = self.duration_labels/MAX_DURATION

        self.data = normalize(self.data)
        self.encoded_pitch_labels = []

        for label in self.pitch_labels:
            self.encoded_pitch_labels.append(self.one_hot_encode_pitch(label))

        self.encoded_pitch_labels = np.array(self.encoded_pitch_labels)        
    
    def one_hot_encode_pitch(self, note):
        one_hot_array = np.zeros((109 - 21,))
        one_hot_array[note - 21] = 1
        return one_hot_array

    def one_hot_encode_vel(self, vel):
        if vel > 0:
            return [0,1]
        else:
            return [1,0]


