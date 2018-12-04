import file_utils as utils
from trainer import ModelCheckpointer
from trainer import tf_model
from utilities import parse_arg
from numpy.random import seed
from datetime import datetime
from os import path
import os
import logging
import random
from trainer.tf_generator import DataGenerator

from tensorflow import set_random_seed
from trainer.logger import NBatchLogger
import numpy as np
from trainer.callbacks import NumbersHistory
from keras.callbacks import EarlyStopping
import tensorflow as tf

seed(1337)
set_random_seed(1337)

date_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
folder_str = 'model-inkml-' + date_str
weights_fname = 'weights_{epoch}.h5'
history_fname = 'history.pkl'
results_fname = 'results.pkl'

start_epoch = int(parse_arg('--start-epoch', 0))
data_base_dir = parse_arg('--data-base-dir', '/Users/balazs/new_data')
model_checkpoint_dir = parse_arg('--model-dir', '/Users/balazs/university/tf_model')
tensorboard_log_dir = parse_arg('--tb', None, required=False)
base_dir = path.join(model_checkpoint_dir, folder_str)
if not path.exists(base_dir):
    os.mkdir(base_dir)

model_weights_file = path.join(model_checkpoint_dir, folder_str, weights_fname)
results_file = path.join(model_checkpoint_dir, folder_str, results_fname)
history_file = path.join(model_checkpoint_dir, folder_str, history_fname)

start_time = datetime.now()
git_hexsha = parse_arg('--git-hexsha', 'NAN')

batch_size = 4
epochs = 30
encoding_vb, decoding_vb = utils.read_pkl(path.join(data_base_dir, "vocabulary.pkl"))

image, truth, _ = zip(*utils.read_pkl(path.join(data_base_dir, "data_train.pkl")))

generator = DataGenerator(image, truth, encoding_vb, batch_size)

input_images = tf.placeholder(tf.float32, shape=(batch_size, None, None, 1), name="input_images")
input_characters = tf.placeholder(tf.int32, shape=(batch_size, None), name="input_characters")
is_training = tf.placeholder(tf.bool, shape=(), name="is_training")

single_image = tf.placeholder(tf.float32, shape=(1, None, None, 1), name="single_input_image")
single_character = tf.placeholder(tf.int32, shape=(1, 1))

logging.debug("Image2Latex: Start create model:", datetime.now().time())
with tf.device('/cpu:0'):
    model = tf_model.Model(len(encoding_vb),
                           encoder_size=256,
                           decoder_units=512,
                           attention_dim=512)
    # Training
    training_output = model.training(input_images, input_characters)

    # Evaluating
    #feature_grid = model.feature_grid(single_image, False)
    #calculate_h0, calculate_c0 = model.calculate_decoder_init(feature_grid)
    #init_h, init_c = model.decoder_init(1)
    #state_h, state_c, output = model.decoder(feature_grid, single_character, init_h, init_c)

logging.debug("Image2Latex: End create model:", datetime.now().time())

y_tensor = tf.placeholder(dtype=tf.int32, shape=(batch_size, None), name="y_labels")
lengts_tensor = tf.placeholder(dtype=tf.int32, shape=(batch_size,), name="lengths")


with tf.name_scope("loss"):
    sequence_masks = tf.sequence_mask(lengts_tensor, dtype=tf.float32)
    loss = tf.contrib.seq2seq.sequence_loss(training_output, y_tensor, sequence_masks)
    tf.summary.scalar("loss", loss)

with tf.name_scope("train"):
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=1.0)
    train = optimizer.minimize(loss)

tf.summary.image("input", input_images, 4)

merged_summary = tf.summary.merge_all()
init = tf.global_variables_initializer()

logging.debug("Image2Latex Start training...")
with tf.Session() as sess:
    writer = None
    if tensorboard_log_dir is not None:
        writer = tf.summary.FileWriter(tensorboard_log_dir)
        writer.add_graph(sess.graph)
    sess.run(init)
    for epoch in range(epochs):
        generator.reset()
        for step in range(int(len(image)/batch_size) + 1):
            image, label, observation, lengths = generator.next_batch()
            dict = {
                input_images: image,
                input_characters: observation,
                lengts_tensor: lengths,
                y_tensor: label
            }
            if writer is not None:
                if step % 2 == 0:
                    s = sess.run(merged_summary, dict)
                    writer.add_summary(s, step)
            loss_val, _ = sess.run([loss, train], feed_dict=dict)
            print(loss_val)


#checkpointer = ModelCheckpointer(filepath=model_weights_file, verbose=1)
#numbers = NumbersHistory(date_str, git_hexsha=git_hexsha)
#logger = NBatchLogger(1)
#stopping = EarlyStopping(patience=2)

#train_len = int(len(x_train)/batch_size)
#val_len = int(len(x_valid)/batch_size)
#epochs = 20
#history = model.fit_generator(training_data, train_len, epochs=epochs, verbose=2,
#                              validation_data=validation_data, validation_steps=val_len,
#                              callbacks=[checkpointer, logger, numbers, stopping], initial_epoch=start_epoch)
#logging.debug(model.metrics_names)
#del history.model#
#utils.write_pkl(history_file, history)
#del numbers.model
#utils.write_pkl(results_file, numbers)
