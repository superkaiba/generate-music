from tensorflow import keras
from tensorflow.keras import layers
from params import NUM_TIMESTEPS, MIN_VECTOR, MAX_VECTOR, RANGE_VECTOR
import mido as md 
import numpy as np

def create_model(possible_labels_length):

    model = keras.Sequential()
    model.add(layers.LSTM(1024, activation='tanh', recurrent_activation='sigmoid', input_shape=(NUM_TIMESTEPS, 3), return_sequences=False))
    #model.add(layers.LSTM(256, activation='tanh', recurrent_activation='sigmoid', input_shape=(NUM_TIMESTEPS, 1), return_sequences=False))
    model.add(layers.Dense(possible_labels_length, activation='softmax'))
    return model

def get_start_of_midi_file(filename):
    notes = clean_midi_file(filename)
    notes_array = np.array(notes)
    return notes_array

def clean_midi_file(filename):
        midi_file = md.MidiFile(filename)
        merged_track = md.merge_tracks([midi_file.tracks[1],midi_file.tracks[2]]) # Merge both piano tracks into one track

        relevant_notes = [x for x in merged_track if (x.type == "note_on" and (x.velocity != 0 or x.time !=0))] # Filter out useless midi messages that have no volume or time value
        notes_to_return = []
        j=0
        for x in relevant_notes:
            if x.velocity > 0:
                velocity = 1
            else:
                velocity = 0
            notes_to_return.append(np.array([x.note, round_to_ten(x.time), velocity]))
        return notes_to_return  
          

def round_to_ten(note_time):
    # if change this function, change POSSIBLE_TIMES param in params.py
    max_time = MAX_VECTOR[1]
    if note_time >= max_time: 
        return max_time
    elif note_time >= 240:
        return note_time - (note_time % 240)
    elif note_time >= 180:
        return 180
    elif note_time >= 120:
        return 120
    elif note_time > 0:
        return note_time - (note_time % 40) + 40
    else:
        return 0 
            
def normalize(note):
    return (note - MIN_VECTOR)/RANGE_VECTOR

def denormalize(note):
    velocity = 0
    if note[2] != 0:
        velocity = 50
    
    note = (note * RANGE_VECTOR) + MIN_VECTOR
    note[2] = velocity

    return note

def test_data_preprocessing(songfile, savefile):
    notes = clean_midi_file(songfile)
    notes = np.array(notes)
    notes = normalize(notes)

    new_midi_file = md.MidiFile()
    track = new_midi_file.add_track()
    for note in notes:
        note = denormalize(note)
        track.append(md.Message('note_on', note=int(note[0]), time=int(note[1]), velocity=int(note[2])))
    new_midi_file.save(savefile)

test_data_preprocessing("train-data/brahms_opus1_1.mid","testsong3.midi")