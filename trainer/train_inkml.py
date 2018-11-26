import file_utils as utils
from trainer import ModelCheckpointer
from trainer import model
from utilities import parse_arg
from numpy.random import seed
from datetime import datetime
from os import path, mkdir
from sklearn.model_selection import train_test_split
import logging

from tensorflow import set_random_seed
from trainer.sequence import predefined_image_sequence_generator
from trainer.logger import NBatchLogger
from trainer.defaults import *
import numpy as np
from trainer.callbacks import NumbersHistory

seed(1337)
set_random_seed(1337)

date_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
folder_str = 'model-inkml-' + date_str
architecture_fname = 'architecture.json'
weights_fname = 'weights_{epoch}.h5'
history_fname = 'history.pkl'
results_fname = 'results.pkl'

start_epoch = int(parse_arg('--start-epoch', 0))
data_base_dir = parse_arg('--data-base-dir', '/Users/balazs/real_data')
model_checkpoint_dir = parse_arg('--model-dir', '/Users/balazs/university/model')
#continue_dir = parse_arg('--continue', default=None, required=False)
base_dir = path.join(model_checkpoint_dir, folder_str)
if not path.exists(base_dir):
    mkdir(base_dir)

model_architecture_file = path.join(model_checkpoint_dir, folder_str, architecture_fname)
model_weights_file = path.join(model_checkpoint_dir, folder_str, weights_fname)
results_file = path.join(model_checkpoint_dir, folder_str, results_fname)
history_file = path.join(model_checkpoint_dir, folder_str, history_fname)

start_time = datetime.now()
git_hexsha = parse_arg('--git-hexsha', 'NAN')
log = "git hash:\t\t\t'" + git_hexsha + "'\n"
log += 'start time:\t\t\t' + str(start_time) + '\n'

batch_size = 2
max_length = 200

vocabulary = utils.read_pkl(path.join(data_base_dir, "vocabulary.pkl"))
vocabulary = vocabulary | {"<start>", "<end>", "^", "_", "\\frac", "{", "}", "\\mbox", "\\to", "\\left"} \
                        | {"\\right", "\\cdots"}
vocabulary = sorted(vocabulary)
vocabulary_maps = create_vocabulary_maps(vocabulary)

images = utils.read_pkl(path.join(data_base_dir, "data_training.pkl"))
x_train, y_train = zip(*images)

images = utils.read_pkl(path.join(data_base_dir, "data_validation.pkl"))
x_valid, y_valid = zip(*images)


#x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2)


# generate data generators
#data_generator = ImageDataGenerator(rotation_range=20,width_shift_range=0.2, height_shift_range=0.2)
training_data = predefined_image_sequence_generator(x_train, y_train, vocabulary_maps[0], None, batch_size)
validation_data = predefined_image_sequence_generator(x_valid, y_valid, vocabulary_maps[0], None, batch_size)

mask = np.zeros(len(vocabulary))
mask[vocabulary_maps[0]['<end>']] = 1


logging.debug("Image2Latex: Start create model:", datetime.now().time())
#with tf.device('/gpu:0'):
model, encoder, decoder = model.create_default(len(vocabulary), mask)

# I don't do this, because I think there are some bugs, when saving RNN with constants
logging.debug("Image2Latex: End create model:", datetime.now().time())

checkpointer = ModelCheckpointer(filepath=model_weights_file, verbose=1)
numbers = NumbersHistory(date_str, git_hexsha=git_hexsha)
logger = NBatchLogger(1)
logging.debug("Image2Latex Start training...")

train_len = 20 #int(len(x_train)/batch_size)
val_len = 10 #int(len(x_valid)/batch_size)
epochs = 1 #10
history = model.fit_generator(training_data, train_len, epochs=epochs, verbose=2,
                              validation_data=validation_data, validation_steps=val_len,
                              callbacks=[checkpointer, logger, numbers], initial_epoch=start_epoch)
end_time = datetime.now()
log += 'end time:\t\t\t' + str(end_time) + '\n'
logging.debug(model.metrics_names)
del history.model
utils.write_pkl(history_file, history)
del numbers.model
utils.write_pkl(results_file, numbers)
