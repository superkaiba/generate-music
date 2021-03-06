# Generate music with a LSTM neural network with attention
Final project for MAIS 202: Accelerated Introduction to Machine Learning bootcamp, hosted by the McGill Artificial Intelligence Society.

Goal: Generate tonal piano music using a LSTM neural network with attention

## Process
Information about model architecture, challenges, data preprocessing and process can be found here: [DevPost](https://devpost.com/software/generate-music)

## Examples
Examples in can be found [here](https://soundcloud.com/thomas-jiralerspong/sets/music-generated-by-neural-network) or in the "examples" folder of the repository.

The start of each example is the starting sequence which was used to generated it (see devpost link above for details). A very high note is played to separate the starting sequence from the music generated by the model.

## Packages
To get necessary packages: pip3 install -r "requirements.txt"

## Training the model as is
- Run `generate_data.py` with default parameters

- Run `train.py`

## Generating music
- Run `generate_data.py` with default parameters if not already done

- Run `generate_music.py`, changing weights_path, midi_output_dir and wav_output_dir as desired

## Making changes to model
- Change model architecture in `model.py -> MusicGenerator -> create_model()`

- Change loss functions or loss weights in `model.py -> MusicGenerator -> train.py`

- Change number of timesteps in `params.py` (requires rerunning `generate_data.py`)

Once changes are made, run `train.py` to train new model

## Running the webapp
- run `generate_composer_data.py`

- run `app.py`

## Dataset
Dataset can be found already separated in 'dataset/train-data' and 'dataset/test-data' folders

A subset of the dataset containing only pieces by Mozart and Haydn (useful for quickly testing models) can be found in 'dataset/smaller-dataset-test' and 'dataset/smaller-dataset-train'. 

Data taken from here: http://www.piano-midi.de/

Train/test split can be done differently, but pretrained weights provided assume current train/test split

## Deliverables
Deliverables for McGill Artificial Intelligence Society's Accelerated Introduction to Machine Learning Bootcamp found in 'Deliverables' folder

## Possible additional experimentation
- Experiment with the use of transformers
- Augment dataset by slightly increasing/decreasing pitch/duration of each note
- Use the [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro) and have velocity as a regression problem instead of a binary classification problem to emulate human performance
