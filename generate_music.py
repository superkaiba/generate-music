from model import MusicGenerator
from params import *

SEQUENCE_LENGTH = 500
WEIGHTS_PATH = "webapp-data/weights.hdf5"
MIDI_OUTPUT_DIR = "static/midi"
WAV_OUTPUT_DIR = "static/audio"
model = MusicGenerator()
model.generate(
            sequence_length=SEQUENCE_LENGTH,
            weights_path=WEIGHTS_PATH,
            generation_data_path=TEST_DATA_FPATH ,
            midi_output_dir=MIDI_OUTPUT_DIR,
            wav_output_dir=WAV_OUTPUT_DIR,
            )