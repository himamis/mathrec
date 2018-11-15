import file_utils as utils
from trainer import ModelCheckpointer
from trainer import model
from utilities import parse_arg
from numpy.random import seed
from datetime import datetime
from os import path, mkdir
from graphics import augment
from xainano_graphics import postprocessor
from trainer.callbacks import EvaluateModel

from tensorflow import set_random_seed
from trainer.sequence import create_default_sequence_generator
from trainer.logger import NBatchLogger
from trainer.defaults import *
import numpy as np

# set seeds so that every trainingsprocess is starting with same weights.
# it is also needed when creating the model and setting weights from a file,
# because there must be some kind of randomness in it, because with not setting
# seeds it generates different output every time. But I don't know where :(

seed(1337)
set_random_seed(1337)

date_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
folder_str = 'model-' + date_str
architecture_fname = 'architecture.json'
weights_fname = 'weights_{epoch}.h5'
result_fname = 'result_log.txt'
history_fname = 'history.pkl'

start_epoch = int(parse_arg('--start-epoch', 0))
data_base_dir = parse_arg('--data-base-dir', '/Users/balazs/university/split')
model_checkpoint_dir = parse_arg('--model-dir', '/Users/balazs/university/model')
background_dir = parse_arg('--background-dir', '/Volumes/SDCard/split_backgrounds_dir')
continue_dir = parse_arg('--continue', default=None, required=False)
base_dir = path.join(model_checkpoint_dir, folder_str)
if not path.exists(base_dir):
    mkdir(base_dir)
data_base_dir = path.join(data_base_dir, 'xainano_images')

model_architecture_file = path.join(model_checkpoint_dir, folder_str, architecture_fname)
model_weights_file = path.join(model_checkpoint_dir, folder_str, weights_fname)
results_file = path.join(model_checkpoint_dir, folder_str, result_fname)
history_file = path.join(model_checkpoint_dir, folder_str, history_fname)

start_time = datetime.now()
log = "git hash:\t\t\t'" + parse_arg('--git-hexsha', 'NAN') + "'\n"
log += 'start time:\t\t\t' + str(start_time) + '\n'

batch_size = 32
max_length = 200

generator = create_generator()
config = create_config()
vocabulary = create_vocabulary(generator, config)
vocabulary_maps = create_vocabulary_maps(vocabulary)
train_token_parser = create_token_parser(path.join(data_base_dir, 'training'))
validation_token_parser = create_token_parser(path.join(data_base_dir, 'validation'))

# generate data generators
train_augmentor = augment.Augmentor(path.join(background_dir, 'training/backgrounds'), path.join(background_dir, 'training/grids'))
validation_augmentor = augment.Augmentor(path.join(background_dir, 'validation/backgrounds'), path.join(background_dir, 'validation/grids'))
post_processor = postprocessor.Postprocessor()
training_data = create_default_sequence_generator(train_token_parser, train_augmentor, post_processor, generator, config, batch_size, vocabulary_maps)
validation_data = create_default_sequence_generator(validation_token_parser, validation_augmentor, post_processor, generator, config, batch_size, vocabulary_maps)

mask = np.zeros(len(vocabulary))
mask[vocabulary_maps[0]['<end>']] = 1


print("Image2Latex:", "Start create model:", datetime.now().time())
model, encoder, decoder = model.create_default(len(vocabulary), mask)
# I don't do this, because I think there are some bugs, when saving RNN with constants
# utils.write_string(model_architecture_file, model.to_json())
print("Image2Latex:", "End create model:", datetime.now().time())
# utils.write_npy(model_weights_file.format(epoch=0), model.get_weights())
if continue_dir is not None and start_epoch != 0 and utils.file_exists(continue_dir):
    model_weights_file = path.join(continue_dir, weights_fname)
    weigths_file = model_weights_file.format(epoch=start_epoch)
    if utils.file_exists(weigths_file):
        print('Start loading weights')
        weights = utils.read_npy(weigths_file)
        model.set_weights(weights)
        print('Weights loaded')

eval = EvaluateModel(encoder, decoder, vocabulary, vocabulary_maps[0], vocabulary_maps[1], validation_data)
checkpointer = ModelCheckpointer(filepath=model_weights_file, verbose=1)
logger = NBatchLogger(1)
print("Image2Latex:", "Start training...")
history = model.fit_generator(training_data, 100, epochs=20, verbose=2,
                              validation_data=validation_data, validation_steps=100,
                              callbacks=[checkpointer, logger, eval], initial_epoch=start_epoch)
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
