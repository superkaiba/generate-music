import mido as md
import numpy as np
import os
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

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
        self.data = keras.utils.normalize(self.data)


    
    def clean_midi_file(self, filename):
        midi_file = md.MidiFile(filename)
        merged_track = md.merge_tracks([midi_file.tracks[1],midi_file.tracks[2]]) # Merge both piano tracks into one track

        relevant_notes = [x for x in merged_track if (x.type == "note_on" and (x.velocity != 0 or x.time != 0))] # Filter out useless midi messages that have no volume or time value
        relevant_notes = [np.array([x.note, x.time]) for x in relevant_notes] # Remove all information from midi messages except time since last note and pitch
        
        return relevant_notes

myData = Data("test")
print(myData.data.shape)
print(myData.labels.shape)
# divby50 = (len(myData.labels) - (len(myData.labels) % 50))
# myData.encoded_data = myData.encoded_data[:divby50,:]
# myData.data = myData.data[:divby50,:]
# myData.data = myData.data.reshape(-1,50,2)
# myData.labels = myData.labels[:divby50,:]
# myData.labels = myData.labels.reshape(-1,50,2)
# print(myData.encoded_data.shape)

# myData.encoded_labels = myData.encoded_labels[:divby50,:]
# myData.encoded_labels = myData.encoded_labels.reshape(-1, 50, len(myData.possible_labels))
# print(myData.encoded_labels)
# print(myData.encoded_labels.shape)
# print(np.where(myData.encoded_labels==1))
# for label in myData.encoded_labels:
#     print(np.where(label==1))
# print(myData.encoded_labels)

model = keras.Sequential()
model.add(layers.LSTM(512, activation='relu', recurrent_activation='sigmoid', input_shape=(50, 2), return_sequences=False))
model.add(layers.Dense(2, activation='relu'))

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(
    optimizer=opt, 
    loss="mse", 
    metrics=['accuracy']
    )

model.summary()
model.fit(myData.data, myData.labels, batch_size=7, epochs=10)
model.save("lstmmodel")
# # model = keras.models.load_model('lstmmodel')
input_array = np.array([36, 2160])
# input_array = myData.one_hot_encode(repr(input_array))
input_array = input_array.reshape(1,1,2)
print(input_array)
print(model.predict(input_array))


# new_midi_file = md.MidiFile()
# track = new_midi_file.add_track()
# new_midi_file.save("new_midi_file.midi")

'''
HOW TO CHOOSE LAYERS:
    - Mostly empirical
    - Look up structures online
    - Tendency to funnel (from more nodes to less)
    - Tendency to use powers of 2
'''

'''
REGULARIZATION TIPS:
    - Data Augmentation (Get more training data)
        ex: - Apply random noise to data (add small amount to each feature)
            - Transform training data (ex: move 1 in MNIST to different part of image)
'''
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