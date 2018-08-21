from file_utils import utils
from trainer import ModelCheckpointer
from trainer import model
import sys
import datetime
from numpy.random import seed

from generator import *
from token_parser import Parser
from xainano_graphics import create_graphics_factory

from tensorflow import set_random_seed
from trainer.sequence import xainano_sequence_generator
from trainer.logger import NBatchLogger
import tensorflow as tf

# set seeds so that every trainingsprocess is starting with same weights.
# it is also needed when creating the model and setting weights from a file,
# because there must be some kind of randomness in it, because with not setting
# seeds it generates different output every time. But I don't know where :(

seed(1337)
set_random_seed(1337)


def get_param(name, default):
    if name in sys.argv:
        i_arg = sys.argv.index(name) + 1
        if i_arg < len(sys.argv):
            print(name + '\t resolved to \t' + sys.argv[i_arg])
            return sys.argv[i_arg]
        else:
            print(name + '\t using default \t' + str(default))
            return default
    else:
        print(name + '\t using default \t' + str(default))
        return default


start_epoch = int(get_param('--start-epoch', 0))
data_base_dir = get_param('--data-base-dir', '/Users/balazs/university/')
model_checkpoint_dir = get_param('--model-dir', '/Users/balazs/university/')
model_architecture_file = model_checkpoint_dir + 'model/architecture.json'
model_weights_file = model_checkpoint_dir + 'model/weights_{epoch}.h5'

embedding_size = 80  # not needed in current version
encoder_size = 256
batch_size = 8

config = Config(".", c(["x"]), None)
generator = random_generator()
# End token
vocabulary = generator.vocabulary(config) | {"<start>", "<end>"}
vocabulary_map = {val: idx for idx, val in enumerate(vocabulary)}
token_parser = Parser(create_graphics_factory(data_base_dir))

# generate data generators
training_data = xainano_sequence_generator(generator, config, token_parser, batch_size, vocabulary_map)
validation_data = xainano_sequence_generator(generator, config, token_parser, batch_size, vocabulary_map)
testing_data = xainano_sequence_generator(generator, config, token_parser, batch_size, vocabulary_map)

with tf.device('/gpu:0'):
    print("Image2Latex:", "Start create model:", datetime.datetime.now().time())
    model, encoder, decoder = model.create(len(vocabulary), embedding_size, encoder_size)
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
