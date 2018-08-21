import file_utils as utils
from trainer import ModelCheckpointer
from trainer import model
from args_parser import parse_arg
import datetime
from numpy.random import seed

from tensorflow import set_random_seed
from trainer.sequence import create_default_sequence_generator
from trainer.logger import NBatchLogger

from trainer.defaults import *

# set seeds so that every trainingsprocess is starting with same weights.
# it is also needed when creating the model and setting weights from a file,
# because there must be some kind of randomness in it, because with not setting
# seeds it generates different output every time. But I don't know where :(

seed(1337)
set_random_seed(1337)


start_epoch = int(parse_arg('--start-epoch', 0))
data_base_dir = parse_arg('--data-base-dir', '/Users/balazs/university/')
model_checkpoint_dir = parse_arg('--model-dir', data_base_dir)
model_architecture_file = model_checkpoint_dir + 'model/architecture.json'
model_weights_file = model_checkpoint_dir + 'model/weights_{epoch}.h5'

batch_size = 8

generator = create_generator()
config = create_config()
vocabulary = create_vocabulary(generator, config)
encoder_vocabulary, decoder_vocabulary = create_vocabulary_maps(vocabulary)
token_parser = create_token_parser(data_base_dir)

# generate data generators
training_data = create_default_sequence_generator(token_parser, generator, config, batch_size, vocabulary_map)
validation_data = create_default_sequence_generator(token_parser, generator, config, batch_size, vocabulary_map)
testing_data = create_default_sequence_generator(token_parser, generator, config, batch_size, vocabulary_map)

print("Image2Latex:", "Start create model:", datetime.datetime.now().time())
model, encoder, decoder = model.create_default(len(vocabulary))
# I don't do this, because I think there are some bugs, when saving RNN with constants
# utils.write_string(model_architecture_file, model.to_json())
print("Image2Latex:", "End create model:", datetime.datetime.now().time())
# utils.write_npy(model_weights_file.format(epoch=0), model.get_weights())

if start_epoch != 0 and utils.file_exists(model_weights_file.format(epoch=start_epoch)):
    print("Image2Latex:", "Start loading weights of epoch", start_epoch)
    weights = utils.read_npy(model_weights_file.format(epoch=start_epoch))
    print("Image2Latex:", "Weights loaded")
    model.set_weights(weights)
    print("Image2Latex:", "Weights set to model")

checkpointer = ModelCheckpointer(filepath=model_weights_file, verbose=1)
logger = NBatchLogger(1)
print("Image2Latex:", "Start training...")
history = model.fit_generator(training_data, 10, epochs=100, verbose=2,
                              validation_data=validation_data, validation_steps=5,
                              callbacks=[checkpointer, logger], initial_epoch=start_epoch)
print("Image2Latex:", history.epoch)
print("Image2Latex:", history.history)
print("Image2Latex:", "Start evaluating...")
losses = model.evaluate_generator(testing_data, 1000)
print(model.metrics_names)
print(losses)
