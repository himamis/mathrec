from trainer import model
from trainer.defaults import *
import file_utils as utils
from utilities import parse_arg
from trainer.logger import NBatchLogger
from os import path
import os
from trainer.sequence import sequence_generator
from numpy.random import seed
from tensorflow import set_random_seed
from keras.layers import Bidirectional, LSTM, Concatenate
from keras import Model
from datetime import datetime
import keras
import keras.backend as K
from keras import models
import numpy as np

import tensorflow as tf

seed(1337)
set_random_seed(1337)

date_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
folder_str = 'decoder_experiment-' + date_str


weights_file = parse_arg('--weights', "/Users/balazs/university/weights_20.h5")
experiment_dir = parse_arg('--experiment_dir', '/Users/balazs/university/decoder_experiment')

output_dir = path.join(experiment_dir, folder_str)
if not path.exists(output_dir):
    os.makedirs(output_dir)

history_file = path.join(output_dir, "history.pkl")

generator = create_generator()
config = create_config()
vocabulary = create_vocabulary(generator, config)
encoding_vb, decoding_vb = create_vocabulary_maps(vocabulary)

batch_size = 32

# generate data generators
data = sequence_generator(generator, config, batch_size, encoding_vb)

print('Start creating model')
default_model, encoder, decoder = model.create_default(len(vocabulary))
print('Model created')
if utils.file_exists(weights_file):
    print('Start loading weights')
    weights = utils.read_npy(weights_file)
    default_model.set_weights(weights)
    print('Weights loaded and set')
else:
    print("Weights file does not exist")
    exit()

for layer in encoder.layers:
    layer.trainable = False

input_tensor = default_model.get_layer("decoder_input_sequences").input
rows = []
for i in range(12):
    row = LSTM(512, return_sequences=True, name="encoder", kernel_initializer="glorot_normal",
                             bias_initializer="zeros")(input_tensor)
    rows.append(row)

feature_grid = Concatenate(axis=1)(rows)
decoder_layer = default_model.layers[35]
softmax_layer = default_model.layers[36]

decoder_output, _, _ = decoder_layer(input_tensor, constants=[feature_grid])
output = softmax_layer(decoder_output)

def get_masked(mask_value, metric):
    mask_value = K.variable(mask_value)
    def masked(y_true, y_pred):
        # find out which timesteps in `y_true` are not the padding character '#'
        mask = K.all(K.equal(y_true, mask_value), axis=-1)
        mask = 1 - K.cast(mask, K.floatx())

        # multiply metric with the mask
        loss = metric(y_true, y_pred) * mask

        # take average w.r.t. the number of unmasked entries
        return K.sum(loss) / K.sum(mask)
    return masked


mask = np.zeros(len(vocabulary))
mask[encoding_vb['<end>']] = 1

masked_categorical_crossentropy = get_masked(mask, K.categorical_crossentropy)
masked_categorical_accuracy = get_masked(mask, keras.metrics.categorical_accuracy)

new_model = Model(input_tensor, output)
new_model.compile(optimizer='adadelta', loss=masked_categorical_crossentropy, metrics=[masked_categorical_accuracy])

logger = NBatchLogger(1)
print("Image2Latex:", "Start training...")
history = new_model.fit_generator(data, 100, epochs=20, verbose=2,
                              validation_data=data, validation_steps=100,
                              callbacks=[logger], initial_epoch=0)
print("Image2Latex:", history.epoch)
print("Image2Latex:", history.history)
print(new_model.metrics_names)
del history.model
utils.write_pkl(history_file, history)

print("done")