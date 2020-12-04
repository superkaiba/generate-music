# Generate music with LSTM neural network with attention
Final project for MAIS 202: Accelerated Introduction to Machine Learning bootcamp, hosted by the McGill Artificial Intelligence Society.

Goal: Generate tonal piano music using a LSTM neural network with attention

Information about architecture can be found (https://devpost.com/software/generate-music "here")
## Packages
To get necessary packages: pip3 install -r "requirements.txt"

## Training the model as is
Run generate_data.py with default parameters
Run train.py 

## Generating music
Run generate_music.py, changing weights_path, midi_output_dir and wav_output_dir as desired

## Making changes to model
Change model architecture in model.py -> MusicGenerator -> create_model()
Change loss functions or loss weights in model.py -> MusicGenerator -> train.py

Change number of timesteps in params.py (requires rerunning generate_data.py)
Once changes are made, run train.py to train new model

## Dataset
Dataset can be found already separated in 'train-data' and 'test-data' folders

Data taken from here: http://www.piano-midi.de/
Train/test split can be done differently, but pretrained weights provided assume current train/test split

## Deliverables
Deliverables for McGill Artificial Intelligence Society's Accelerated Introduction to Machine Learning Bootcamp found in 'Deliverables' folder