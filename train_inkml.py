import file_utils as utils
from trainer import ModelCheckpointer
from trainer import model
from utilities import parse_arg
from numpy.random import seed
from datetime import datetime
from os import path, mkdir
from sklearn.model_selection import train_test_split
from graphics import augment
from xainano_graphics import postprocessor
from trainer.callbacks import EvaluateModel
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from tensorflow import set_random_seed
from trainer.sequence import predefined_image_sequence_generator
from trainer.logger import NBatchLogger
from trainer.defaults import *
import numpy as np
import pickle

seed(1337)
set_random_seed(1337)

date_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
folder_str = 'model-inkml-' + date_str
architecture_fname = 'architecture.json'
weights_fname = 'weights_{epoch}.h5'
result_fname = 'result_log.txt'
history_fname = 'history.pkl'

start_epoch = int(parse_arg('--start-epoch', 0))
data_base_dir = parse_arg('--data-base-dir', '/Users/balazs/')
model_checkpoint_dir = parse_arg('--model-dir', '/Users/balazs/university/model')
#background_dir = parse_arg('--background-dir', '/Volumes/SDCard/split_backgrounds_dir')
#continue_dir = parse_arg('--continue', default=None, required=False)
base_dir = path.join(model_checkpoint_dir, folder_str)
if not path.exists(base_dir):
    mkdir(base_dir)
#data_base_dir = path.join(data_base_dir, 'xainano_images')

model_architecture_file = path.join(model_checkpoint_dir, folder_str, architecture_fname)
model_weights_file = path.join(model_checkpoint_dir, folder_str, weights_fname)
results_file = path.join(model_checkpoint_dir, folder_str, result_fname)
history_file = path.join(model_checkpoint_dir, folder_str, history_fname)

start_time = datetime.now()
log = "git hash:\t\t\t'" + parse_arg('--git-hexsha', 'NAN') + "'\n"
log += 'start time:\t\t\t' + str(start_time) + '\n'

batch_size = 32
max_length = 200

vocabulary = pickle.load(open(path.join(data_base_dir, "vocabulary.pkl"), "rb"))
vocabulary = vocabulary | {"<start>", "<end>", "^", "_", "\\frac", "{", "}", "\\mbox", "\\to", "\\left"} \
                        | {"\\right", "\\cdots"}
vocabulary = sorted(vocabulary)
vocabulary_maps = create_vocabulary_maps(vocabulary)

images = pickle.load(open(path.join(data_base_dir, "images_train.pkl"), "rb"))
x, y = zip(*images)

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2)


# generate data generators
#data_generator = ImageDataGenerator(rotation_range=20,width_shift_range=0.2, height_shift_range=0.2)
training_data = predefined_image_sequence_generator(x_train, y_train, vocabulary_maps[0], None, batch_size)
validation_data = predefined_image_sequence_generator(x_valid, y_valid, vocabulary_maps[0], None, batch_size)

mask = np.zeros(len(vocabulary))
mask[vocabulary_maps[0]['<end>']] = 1


print("Image2Latex:", "Start create model:", datetime.now().time())
model, encoder, decoder = model.create_default(len(vocabulary), mask)
# I don't do this, because I think there are some bugs, when saving RNN with constants
print("Image2Latex:", "End create model:", datetime.now().time())

#eval = EvaluateModel(encoder, decoder, vocabulary, vocabulary_maps[0], vocabulary_maps[1], validation_data)
checkpointer = ModelCheckpointer(filepath=model_weights_file, verbose=1)
logger = NBatchLogger(1)
print("Image2Latex:", "Start training...")
history = model.fit_generator(training_data, len(x_train), epochs=10, verbose=2,
                              validation_data=validation_data, validation_steps=len(x_valid),
                              callbacks=[checkpointer, logger], initial_epoch=start_epoch)
end_time = datetime.now()
log += 'end time:\t\t\t' + str(end_time) + '\n'
print("Image2Latex:", history.epoch)
print("Image2Latex:", history.history)
print("Image2Latex:", "Start evaluating...")
#losses = model.evaluate_generator(testing_data, 1000)
#print(losses)
print(model.metrics_names)
#log += 'losses:\n'
#log += str(losses)
#utils.write_string(results_file, log)
del history.model
utils.write_pkl(history_file, history)
