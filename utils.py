from tensorflow import keras
from tensorflow.keras import layers
from params import *
import mido as md 
import numpy as np
import keras_self_attention
from keras_self_attention import SeqWeightedAttention, SeqSelfAttention

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
        notes_to_return.append(np.array([x.note, cutoff(x.time), velocity]))
    return notes_to_return  
          
def cutoff(note_time):
    if note_time > MAX_DURATION:
        return MAX_DURATION
    else:
        return note_time
            
def normalize(note):
    return (note - MIN_VECTOR)/RANGE_VECTOR

def denormalize(note):
    note = (note * RANGE_VECTOR) + MIN_VECTOR
    
    velocity = 0
    if note[2] != 0:
        velocity = 50

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

#test_data_preprocessing("train-data/brahms_opus1_1.mid","testsong3.midi")
# model = create_model(1408)
# model.save("attention-model")