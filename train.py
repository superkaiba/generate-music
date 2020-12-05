from model import MusicGenerator
import tensorflow as tf 
tf.compat.v1.disable_eager_execution()
model = MusicGenerator()
model.train()