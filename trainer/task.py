import file_utils as utils
from trainer import ModelCheckpointer
from trainer import model_new
from args_parser import parse_arg
import datetime
from numpy.random import seed
import numpy as np

from tensorflow import set_random_seed
from trainer.sequence import create_default_sequence_generator
from trainer.logger import NBatchLogger

from keras.callbacks import LambdaCallback

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

batch_size = 32

generator = create_generator()
config = create_config()
vocabulary = create_vocabulary(generator, config)
vocabulary_maps = create_vocabulary_maps(vocabulary)
token_parser = create_token_parser(data_base_dir)

# generate data generators
training_data = create_default_sequence_generator(token_parser, generator, config, batch_size, vocabulary_maps)
validation_data = create_default_sequence_generator(token_parser, generator, config, batch_size, vocabulary_maps)
testing_data = create_default_sequence_generator(token_parser, generator, config, batch_size, vocabulary_maps)
callback_data = create_default_sequence_generator(token_parser, generator, config, 1, vocabulary_maps)

print("Image2Latex:", "Start create model:", datetime.datetime.now().time())
model, encoder, decoder = model_new.create_default(len(vocabulary))
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


# Function to display the target and prediciton
def testmodel(epoch, logs):
    predx, predy = next(callback_data)

    print("Testing model")
    predout = model.predict(predx,batch_size=1)

    print("Target: ")
    for i in range(0, batch_size):
        seq = []
        for j in range(0, predy[i].shape[0]):
            val = np.argmax(predy[i][j])
            seq.append(vocabulary_maps[1][val])
        print(seq)
        print("\n")

    print("Prediction: ")
    for i in range(0, batch_size):
        seq = []
        for j in range(0, predout[i].shape[0]):
            val = np.argmax(predout[i][j])
            seq.append(vocabulary_maps[1][val])
        print(seq)
        print("\n")


# Callback to display the target and prediciton
testmodelcb = LambdaCallback(on_batch_end=testmodel)

print("Image2Latex:", "Start training...")
history = model.fit_generator(training_data, 10, epochs=100, verbose=2,
                              validation_data=validation_data, validation_steps=5,
                              callbacks=[checkpointer, logger, testmodelcb], initial_epoch=start_epoch)
print("Image2Latex:", history.epoch)
print("Image2Latex:", history.history)
print("Image2Latex:", "Start evaluating...")
losses = model.evaluate_generator(testing_data, 1000)
print(model.metrics_names)
print(losses)
