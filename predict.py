from trainer import model
from trainer.defaults import *
from trainer.sequence import create_default_sequence_generator
import file_utils as utils
import numpy as np
from utilities import parse_arg

data_base_dir = parse_arg('--data-base-dir', '/Users/balazs/university')
weights_file = parse_arg('--weights', "/Users/balazs/university/mathrec/weights_15.h5")
#if weights_file is None:
#    print('Enter base dir:')
#    weights_file = input()


generator = create_generator()
token_parser = create_token_parser(data_base_dir)
config = create_config()
vocabulary = create_vocabulary(generator, config)
vocabulary_maps = create_vocabulary_maps(vocabulary)

print('Start creating model')
model, encoder, decoder = model.create_default(len(vocabulary))
print('Model created')
if utils.file_exists(weights_file):
    print('Start loading weights')
    weights = utils.read_npy(weights_file)
    model.set_weights(weights)
    print('Weights loaded and set')
else:
    print("Weights file does not exist")
    exit()

max_length = 100

def evaluate(image):
    print("Testing model")
    print("Encoding data")
    input_image = np.expand_dims(image, 0)
    feature_grid = encoder.predict(input_image)

    #print("Expected target")
    #target_sentence = [vocabulary_maps[1][np.argmax(char)] for char in predy[0]]
    #print(target_sentence)
    #print("\n")

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
