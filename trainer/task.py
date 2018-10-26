import file_utils as utils
from trainer import ModelCheckpointer
from trainer import model
from utilities import parse_arg
from numpy.random import seed
import numpy as np
from datetime import datetime
from os import path, mkdir
from graphics import augment
from xainano_graphics import postprocessor

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

date_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
folder_str = 'model-' + date_str
architecture_fname = 'architecture.json'
weights_fname = 'weights_{epoch}.h5'
result_fname = 'result_log.txt'
history_fname = 'history.pkl'

start_epoch = int(parse_arg('--start-epoch', 0))
data_base_dir = parse_arg('--data-base-dir', '/Users/balazs/university/data')
model_checkpoint_dir = parse_arg('--model-dir', data_base_dir)
background_dir = parse_arg('--background-dir', data_base_dir)
base_dir = path.join(model_checkpoint_dir, folder_str)
if not path.exists(base_dir):
    mkdir(base_dir)

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
token_parser = create_token_parser(data_base_dir)

# generate data generators
augmentor = augment.Augmentor(background_dir)
post_processor = postprocessor.Postprocessor()
training_data = create_default_sequence_generator(token_parser, augmentor, post_processor, generator, config, batch_size, vocabulary_maps)
validation_data = create_default_sequence_generator(token_parser, augmentor, post_processor, generator, config, batch_size, vocabulary_maps)
testing_data = create_default_sequence_generator(token_parser, augmentor, post_processor, generator, config, batch_size, vocabulary_maps)
callback_data = create_default_sequence_generator(token_parser, augmentor, post_processor, generator, config, 1, vocabulary_maps)


print("Image2Latex:", "Start create model:", datetime.now().time())
model, encoder, decoder = model.create_default(len(vocabulary))
# I don't do this, because I think there are some bugs, when saving RNN with constants
# utils.write_string(model_architecture_file, model.to_json())
print("Image2Latex:", "End create model:", datetime.now().time())
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
    #predx, predy = next(callback_data)
    predx, predy = (None, None)
    print("Testing model")
    print("Encoding data")
    feature_grid = encoder.predict(predx[0])

    print("Expected target")
    target_sentence = [vocabulary_maps[1][np.argmax(char)] for char in predy[0]]
    print(target_sentence)
    print("\n")

    print("Decoding target")
    sequence = np.zeros((1, 1, len(vocabulary)), dtype="float32")
    sequence[0, 0, vocabulary_maps[0]["<start>"]] = 1.

    h = np.zeros((1, 256 * 2), dtype="float32")
    c = np.zeros((1, 256 * 2), dtype="float32")
    states = [h, c]

    decoded_sentence = ""
    while True:
        output, h, c = decoder.predict([feature_grid, sequence] + states)

        # Sample token
        sampled_token_index = np.argmax(output[0, -1, :])
        sampled_char = vocabulary_maps[1][sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: hit max length, or find stop character
        if sampled_char == "<end>" or len(decoded_sentence) > max_length:
            break

        # Update sequence
        sequence = np.zeros((1, 1, len(vocabulary)), dtype="float32")
        sequence[0, 0, sampled_token_index] = 1.

        states = [h, c]

    print("Prediction")
    print(decoded_sentence)
    print("\n")


# Callback to display the target and prediciton
#testmodelcb = LambdaCallback(on_epoch_end=testmodel)

print("Image2Latex:", "Start training...")
history = model.fit_generator(training_data, 100, epochs=10, verbose=2,
                              validation_data=validation_data, validation_steps=100,
                              callbacks=[checkpointer, logger], initial_epoch=start_epoch)
end_time = datetime.now()
log += 'end time:\t\t\t' + str(end_time) + '\n'
print("Image2Latex:", history.epoch)
print("Image2Latex:", history.history)
print("Image2Latex:", "Start evaluating...")
losses = model.evaluate_generator(testing_data, 1000)
print(model.metrics_names)
print(losses)
log += 'losses:\n'
log += str(losses)
utils.write_string(results_file, log)
del history.model
utils.write_pkl(history_file, history)
