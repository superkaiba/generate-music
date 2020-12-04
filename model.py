from tensorflow import keras, math
from tensorflow.keras import layers
import tensorflow as tf

import keras_self_attention
from keras_self_attention import SeqWeightedAttention, SeqSelfAttention

import os
from random import randrange

import mido as md  
from params import *
from utils import *
from midi2audio import FluidSynth

class MusicGenerator:
    def __init__(self, lstm_units=512, model_name="music_generator"):
        self.lstm_units = lstm_units
        self.model_name = model_name
        self.model = self.create_model()

    def create_model(self):

        inputs = layers.Input(shape=(NUM_TIMESTEPS, 3))
        x = SeqSelfAttention(attention_activation='tanh')(inputs)

        hidden_states = layers.Bidirectional(layers.LSTM(
            self.lstm_units, 
            dropout=0.2,
            recurrent_dropout=0.2,
            activation='tanh', 
            recurrent_activation='sigmoid', 
            input_shape=(NUM_TIMESTEPS, 3), 
            return_sequences=True))(x)

        last_hidden_state = hidden_states[:,NUM_TIMESTEPS - 1,:]
        last_hidden_state = layers.Reshape((1, self.lstm_units * 2))(last_hidden_state)

        context_vector = layers.AdditiveAttention()([last_hidden_state, hidden_states, hidden_states])
        context_vector = layers.Reshape((self.lstm_units * 2,))(context_vector)

        last_hidden_state = layers.Reshape((self.lstm_units * 2,))(last_hidden_state)

        x = layers.Concatenate(axis=-1)([last_hidden_state, context_vector])
        x = layers.Dropout(0.2)(x)
        
        vel_output = layers.Dense(units=2, activation='softmax', name='vel_out')(x)
        x = layers.Concatenate(axis=-1)([x, vel_output])

        duration_output = layers.Dense(units=1, activation='sigmoid', name='duration_out')(x)
        x = layers.Concatenate(axis=-1)([x, duration_output])

        pitch_outputs = layers.Dense(units=88, activation = 'softmax', name='pitch_out')(x)
        model = keras.Model(inputs=inputs, outputs=[pitch_outputs, duration_output, vel_output])

        return model
    # def create_model(self):
    #     inputs = layers.Input(shape=(NUM_TIMESTEPS, 3))
    #     x = SeqSelfAttention(attention_activation='tanh')(inputs)
    #     hidden_states = layers.LSTM(
    #         1024, 
    #         activation='tanh', 
    #         recurrent_activation='sigmoid', 
    #         input_shape=(NUM_TIMESTEPS, 3), 
    #         return_sequences=True)(x)

    #     last_hidden_state = hidden_states[:,NUM_TIMESTEPS - 1,:]
    #     last_hidden_state = layers.Reshape((1, 1024))(last_hidden_state)

    #     context_vector = layers.AdditiveAttention()([last_hidden_state, hidden_states, hidden_states])

    #     context_vector = layers.Reshape((1024,))(context_vector)
    #     last_hidden_state = layers.Reshape((1024,))(last_hidden_state)

    #     x = layers.Concatenate(axis=-1)([last_hidden_state, context_vector])
    #     vel_output = layers.Dense(units=2, activation='softmax', name='vel_out')(x)
    #     x = layers.Concatenate(axis=-1)([x, vel_output])

    #     duration_output = layers.Dense(units=1, activation='sigmoid', name='duration_out')(x)
    #     x = layers.Concatenate(axis=-1)([x, duration_output])

    #     pitch_outputs = layers.Dense(units=88, activation = 'softmax', name='pitch_out')(x)
    #     model = keras.Model(inputs=inputs, outputs=[pitch_outputs, duration_output, vel_output])
    #     return model

    def train(self, epochs=50, lr=0.001, initial_epoch=0, train_data_fpath=TRAIN_DATA_FPATH, pitch_labels_fpath=PITCH_LABELS_FPATH, duration_labels_fpath=DURATION_LABELS_FPATH, vel_labels_fpath=VEL_LABELS_FPATH):
        '''Trains model with specified parameters using data from specified filepaths
        '''
        data = np.load(train_data_fpath)
        pitch_labels = np.load(pitch_labels_fpath)
        duration_labels = np.load(duration_labels_fpath)
        vel_labels = np.load(vel_labels_fpath)

        os.mkdir(self.model_name)

        filepath = self.model_name + "/weights-epoch-{epoch:02d}-loss-{loss:.4f}-accuracy-{pitch_out_accuracy:.4f}.hdf5" 
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath, monitor='loss', 
            verbose=0,        
            save_best_only=True,        
            mode='min'
        )  

        csvlogger = keras.callbacks.CSVLogger(f"{self.model_name}/epoch-results.csv",append=True)
        myScheduler = keras.callbacks.LearningRateScheduler(self.scheduler)

        opt = keras.optimizers.Adam(learning_rate=0.001)


        self.model.compile(
            optimizer=opt, 

            loss={
                "pitch_out":"categorical_crossentropy",
                "duration_out":"mse",
                "vel_out":"binary_crossentropy"
                }, 
        # Change loss weights here if want more precision for duration/pitch/velocity
            loss_weights={
                "pitch_out":2,
                "duration_out":1,
                "vel_out":1
            },

            metrics={
                'pitch_out':'accuracy',
                },
            )
        self.model.build(input_shape=data.shape)
        self.model.summary()

        self.model.save(f"{self.model_name}/saved_model_configuration")

        self.model.fit(x=data,  y={'pitch_out': pitch_labels, 'duration_out': duration_labels, 'vel_out': vel_labels}, batch_size=64, epochs=50, callbacks=[checkpoint, csvlogger, myScheduler], initial_epoch=initial_epoch)
             
    def generate(self, sequence_length, weights_path, generation_data_path, midi_output_dir=".", wav_output_dir=".", random_index=None, include_initial_sequence=True):
        
        generation_data = np.load(generation_data_path)
        self.model.build(input_shape=generation_data.shape)
        self.model.load_weights(weights_path)

        if random_index == None:
            random_index = randrange(0,len(generation_data))

        sequence = generation_data[random_index].reshape(1, NUM_TIMESTEPS, 3)

        new_midi_file = md.MidiFile()
        track = new_midi_file.add_track()
        if include_initial_sequence:
            for i in range(NUM_TIMESTEPS):
                note = denormalize(sequence[0][i])
                track.append(md.Message('note_on', note=int(note[0]), time=int(note[1]), velocity=int(note[2])))
            track.append(md.Message('note_on', note=108, time=0, velocity=100)) # Indicate end of initial sequence

        note_range = range(21,109)
        vel_range = [0,1]

        for i in range(sequence_length):
            new_note_pitch, new_note_duration, new_note_vel = self.model.predict(sequence)
            new_note_pitch = note_range[np.argmax(new_note_pitch)]

            denormalized_new_note_duration = self.round_down(new_note_duration * MAX_DURATION)
            new_note_vel = vel_range[np.argmax(new_note_vel)]
            new_note = [new_note_pitch, int(denormalized_new_note_duration), new_note_vel]
            
            sequence = np.append(sequence[:,1:], normalize(new_note)).reshape(1,NUM_TIMESTEPS,3)

            velocity = 0
            if new_note_vel != 0:
                velocity = OUTPUT_VEL
            
            track.append(md.Message('note_on', note=new_note[0], time=int(new_note[1]), velocity = velocity))
            print(i)
        data_path = generation_data_path[5:-4]
        midi_name = f"{midi_output_dir}/{data_path}-random-index-{random_index}.midi"
        audio_name = f"{wav_output_dir}/{data_path}-random-index-{random_index}.wav"
        new_midi_file.save(midi_name)
        FluidSynth().midi_to_audio(midi_name, audio_name)
        return audio_name
        
    def round_down(self, duration):
        if duration > 720:
            return 720
        elif duration < 40:
            return 0
        else:
            return duration
    
    def scheduler(self, epoch, lr):
        if epoch < 5:
            return lr
        else:
            return lr * math.exp(-0.1)

