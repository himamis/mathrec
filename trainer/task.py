import file_utils as utils
from trainer import ModelCheckpointer
from trainer import model
from utilities import parse_arg
import datetime
from numpy.random import seed
import numpy as np
from datetime import datetime
from os import path
from graphics import augment

from tensorflow import set_random_seed
from trainer.sequence import create_default_sequence_generator, image_sequencer
from trainer.logger import NBatchLogger

from keras.callbacks import LambdaCallback, LearningRateScheduler

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
#vocabulary_maps = create_vocabulary_maps(vocabulary)
#token_parser = create_token_parser(data_base_dir)

def formula_string_to_tokens(formula):
    full_tokens = ["\\frac", "\\alpha", "\\beta", "\\theta", "\\sigma", "\\pi"]
    index = 0
    tokens = []

    while index < len(formula):
        found = False
        for full_token in full_tokens:
            if formula[index:].startswith(full_token):
                tokens.append(full_token)
                index += len(full_token)
                found = True
                break
        if not found:
            tokens.append(formula[index])
            index += 1

    return tokens


def image_map(base):
    file = open(path.join(base, "data.txt"), "r")
    lines = file.readlines()
    file.close()
    array = []
    for line in lines:
        fname, truth = line.replace("\n", "").split("\t")

        array.append((fname, formula_string_to_tokens(truth)))
    return array

def extend_vocabulary(image_map, vocabulary):
    for (_, formula)  in image_map:
        for c in formula:
            if c not in vocabulary:
                vocabulary |= {c}
    return vocabulary

def combine_generators(generator1, generator2):
    while True:
        yield next(generator1)
        yield next(generator2)


xainano_path = path.join(data_base_dir, "xainano_images")
inkml_path = path.join(data_base_dir, "TC11_package")
xainano_image_map = image_map(xainano_path)
inkml_image_path = image_map(inkml_path)

vocabulary = extend_vocabulary(inkml_image_path, vocabulary)
vocabulary_maps = create_vocabulary_maps(vocabulary)

# generate data generators
#training_data = create_default_sequence_generator(token_parser, generator, config, batch_size, vocabulary_maps)
#validation_data = create_default_sequence_generator(token_parser, generator, config, batch_size, vocabulary_maps)
#testing_data = create_default_sequence_generator(token_parser, generator, config, batch_size, vocabulary_maps)
#callback_data = create_default_sequence_generator(token_parser, generator, config, 1, vocabulary_maps)
augmentor = augment.Augmentor(path.join(data_base_dir, "backgrounds"))
training_data1 = image_sequencer(batch_size, xainano_image_map, xainano_path, vocabulary_maps[0], augmentor, split=(0, 80))
training_data2 = image_sequencer(batch_size, inkml_image_path, inkml_path, vocabulary_maps[0], augmentor, split=(0, 80))
training_data = combine_generators(training_data1, training_data2)
validation_data1 = image_sequencer(batch_size, xainano_image_map, xainano_path, vocabulary_maps[0], augmentor, split=(80, 90))
validation_data2 = image_sequencer(batch_size, inkml_image_path, inkml_path, vocabulary_maps[0], augmentor, split=(80, 90))
validation_data = combine_generators(validation_data1, validation_data2)
testing_data1 = image_sequencer(batch_size, xainano_image_map, xainano_path, vocabulary_maps[0], augmentor, split=(90, 100))
testing_data2 = image_sequencer(batch_size, inkml_image_path, inkml_path, vocabulary_maps[0], augmentor, split=(90, 100))
testing_data = combine_generators(testing_data1, testing_data2)


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
utils.write_pkl(history_file, history)
