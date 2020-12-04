# Generate music with LSTM neural network with attention
Final project for MAIS 202: Accelerated Introduction to Machine Learning bootcamp, hosted by the McGill Artificial Intelligence Society.

Goal: Generate tonal piano music using a LSTM neural network with attention

Information about model architecture, challenges, data preprocessing and process can be found [here](https://devpost.com/software/generate-music)
## Packages
To get necessary packages: pip3 install -r "requirements.txt"

## Training the model as is
Run generate_data.py with default parameters

Run train.py 

## Generating music
Run generate_data.py with default parameters if not already done

Run generate_music.py, changing weights_path, midi_output_dir and wav_output_dir as desired

## Making changes to model
Change model architecture in model.py -> MusicGenerator -> create_model()

Change loss functions or loss weights in model.py -> MusicGenerator -> train.py

Change number of timesteps in params.py (requires rerunning generate_data.py)

Once changes are made, run train.py to train new model

## Running the webapp
run generate_composer_data.py

run app.py

## Deliverables
Deliverables for McGill Artificial Intelligence Society's Accelerated Introduction to Machine Learning Bootcamp found in 'Deliverables' folder
