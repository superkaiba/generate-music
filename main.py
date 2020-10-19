import mido as md
import numpy as np
import os
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers

class Data():

    def __init__(self, dirname):
        self.data = []
        self.labels = []
        for file in os.listdir(dirname):
            notes, nextnotes = self.clean_midi_file(os.path.join(dirname, file))
            self.data.append(np.array(notes))
            self.labels.append(np.array(nextnotes))
            
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
    
    def clean_midi_file(self, filename):
        midi_file = md.MidiFile(filename)
        merged_track = md.merge_tracks([midi_file.tracks[1],midi_file.tracks[2]]) # Merge both piano tracks into one track

        relevant_notes = [x for x in merged_track if (x.type == "note_on" and (x.velocity != 0 or x.time != 0))] # Filter out useless midi messages that have no volume or time value
        relevant_notes = [np.array([x.note, x.time]) for x in relevant_notes] # Remove all information from midi messages except time since last note and pitch
        
        nextnotes = relevant_notes[1:] # Label each element with the next element by removing first element from array
        relevant_notes = relevant_notes[:-1] # Remove last element from data array since last element doesn't have a next note
        
        return relevant_notes, nextnotes  

myData = Data("test")

print(myData.data)
print(myData.labels)
print(myData.data[0].shape)
print(myData.labels[0].shape)

# possible_notes = np.unique(clean_data[:,0])
# possible_times = np.unique(clean_data[:,1])
# possible_labels = np.array(np.meshgrid(possible_notes, possible_times)).T.reshape(-1,2)

# model.summary()
# new_midi_file = md.MidiFile()
# track = new_midi_file.add_track()
# new_midi_file.save("new_midi_file.midi")

''' 

POSSIBLE DATASETS: 
    - http://www.kunstderfuge.com/bach.htm
    - https://www.kaggle.com/soumikrakshit/classical-music-midi
    - LOOK FOR OTHERS
NOTES: 

    - Time attribute of note_on message is time SINCE last note
    - Therefore if time attribute = 0 then note is simultaneous with last note
    - First note of file --> Time attribute is time since start of "piece"
    - Maybe use website (https://solmire.com/miditosheetmusic/) 
      instead of MuseScore to convert from MIDI to score because MuseScore puts weird silences and weird note lengths (or change MuseScore settings)
    - Maybe try to generalize for multiple tracks/instruments
    - Track 1 (Index 1 not index 0) has notes played by right hand
    - Track 2 (Index 2 not index 1) has notes played by left hand
    - Sometimes in chord 2 of same note but with different velocity, maybe try to remove them in preprocessing of data to not confuse model
    - Maybe find cleaner dataset with cleaner note lengths instead of MAGENTA one which is human performances... Might have to combine multiple.
      Or just use MAGENTA dataset and find way to convert human performance to more standard~/use website to convert final machine learning product into something more readable
      Or don't care about converting to score at all.

IDEAS:
    - Choose specific style? Train model to recreate music of certain style/period, or multiple different styles/periods.

NOTES ON LSTM: 
    - Each datapoint is 1 note, label is next note? So has to predict next note according to current note.
    - Loss function could be based on distance to next note, whether is a consonant interval, whether is same octave???

STEPS:
    - Get all notes onto 1 track (MIDI type 1 to MIDI type 0)
    - Put all data from midi files into array, removing velocity and message type, so only have [note, time] integer pair
    - Get all possible velocities and all possible note values by going through array and getting all unique values
    - Label each note with next note 
    - Create possible labels as all combinations of note values and time values
    - Feed data and all possible labels to LSTM model, TRAIN
    - HAVE TO MAKE SURE THAT LSTM "forgets" all previous data when new piece starts, if possible..?
    - Use trained model with arbitrary starting note (or multiple if possible) as first input, then each predicted note as next input

'''