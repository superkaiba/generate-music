# generate-music
Final project for MAIS 202: Accelerated Introduction to Machine Learning bootcamp, hosted by the McGill Artificial Intelligence Society.

Goal: Generate tonal piano music using machine learning. 

## Packages
To get necessary packages: pip3 install -r "requirements.txt"

## Project structure
Change model architecture in utils.py -> create_model()

Use Data.py to generate new .npy files to train with

Use train.py to train model from .npy files (train data, possible labels, encoded labels)

Use params.py to change general parameters such as number of timesteps for model, or possible durations for notes

## Dataset
Large dataset can be found in 'train-data' and 'test-data' folders

Smaller dataset to test out model can be found in 'smaller-dataset-train' and 'smaller-dataset-test' folders (Just Mozart and Haydn)

Data taken from here: http://www.piano-midi.de/

## Deliverables
Deliverables for McGill Artificial Intelligence Society's Accelerated Introduction to Machine Learning Bootcamp found in 'Deliverables' folder