from Data import Data
import numpy as np
import os
from params import *

TRAIN_DATA_FOLDER = "dataset/train-data"
TEST_DATA_FOLDER = "dataset/test-data"

os.mkdir(PROCESSED_DATA_DIRNAME)

train_data = Data(TRAIN_DATA_FOLDER)

np.save(PITCH_LABELS_FPATH, train_data.encoded_pitch_labels)
np.save(DURATION_LABELS_FPATH, train_data.duration_labels)
np.save(VEL_LABELS_FPATH, train_data.encoded_vel_labels)
np.save(TRAIN_DATA_FPATH, train_data.data)

test_data = Data(TEST_DATA_FOLDER)
np.save(TEST_DATA_FPATH, test_data.data)