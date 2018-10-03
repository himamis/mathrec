from trainer import model
from trainer.defaults import *
from trainer.sequence import create_default_sequence_generator
import file_utils as utils
import numpy as np
import cv2
from functools import reduce
from numpy.random import seed
from tensorflow import set_random_seed
from utilities import parse_arg

seed(1336)
set_random_seed(1336)

data_base_dir = parse_arg('--data-base-dir', None)
if data_base_dir is None:
    print('Enter base dir:')
    data_base_dir = input()

vocabulary_set = create_vocabulary()
weights_file = data_base_dir + 'model/weights_{epoch}.h5'
encoder_vocabulary, decoder_vocabulary = create_vocabulary_maps(vocabulary_set)
generator = create_generator()

token_parser = create_token_parser(data_base_dir)
sequence = create_default_sequence_generator(token_parser, batch_size=1)

print('Vocabulary read. Size is', len(encoder_vocabulary))
print('Start creating model')
model, encoder, decoder = model.create_default(len(encoder_vocabulary))
print('Model created')
for epoch in reversed(range(10)):
    file = weights_file.format(epoch=epoch + 1)
    if utils.file_exists(file):
        print('Start loading weights', epoch + 1)
        weights = utils.read_npy(file)
        model.set_weights(weights)
        print('Weights loaded and set')
        break


for inputs, output in sequence:
    inp = inputs[0]
    image = inp[0:1, :]

    #s = reduce((lambda a, b: a + " " + b), inputs[1])
    #print("Token sequence is: " + s)
    cv2.imshow('image', image[0])

    encoded_image = encoder.predict(image)

    seq = np.array([[encoder_vocabulary['<start>']]])

    while True:
        y = decoder.predict([seq, encoded_image])
        max = np.argmax(y)
        token = decoder_vocabulary[max]
        print(decoder_vocabulary[np.argmax(y)])
        if token == '<end>':
            break
        seq = np.array([[max]])
    print('')
    cv2.destroyAllWindows()