from trainer import model
from trainer.defaults import *
from trainer.sequence import create_default_sequence_generator
import file_utils as utils
import numpy as np
import cv2
from functools import reduce
from numpy.random import seed
from tensorflow import set_random_seed

seed(1337)
set_random_seed(1337)
model.create_default()

print('Enter base dir:')
data_base_dir = input()
vocabulary_set = create_vocabulary()
weights_file = data_base_dir + 'model/weights_{epoch}.h5'
vocabulary = create_vocabulary_map()
generator = create_generator()

token_parser = create_token_parser(data_base_dir)
sequence = create_default_sequence_generator(token_parser)

print('Vocabulary read. Size is', len(vocabulary))
print('Start creating model')
model = model.create_default(len(vocabulary), True)
print('Model created')
for epoch in reversed(range(10)):
    file = weights_file.format(epoch=epoch + 1)
    if utils.file_exists(file):
        print('Start loading weights', epoch + 1)
        weights = utils.read_npy(file)
        model.set_weights(weights)
        print('Weights loaded and set')
        break

while True:
    [inputs, output] = sequence()[0]

    s = reduce((lambda a, b: a + " " + b), inputs[1])
    print("Token sequence is: " + s)
    cv2.imshow('image', inputs[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    y = model.predict(inputs, batch_size=1)
    for seq in y:
        outp = ''
        for tok in seq:
            outp += vocabulary[np.argmax(tok)] + ' '
        print(outp)
    print('')