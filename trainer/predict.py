from trainer import model
from file_utils import utils
import numpy as np
from trainer.defaults import *
import cv2

# set seeds so that every trainingsprocess is starting with same weights.
# it is also needed when creating the model and setting weights from a file, 
# because there must be some kind of randomness in it, because with not setting
# seeds it generates different output every time. But I don't know where :(
from numpy.random import seed
from tensorflow import set_random_seed
from trainer.sequence import create_default_sequence_generator
from functools import reduce

seed(1337)
set_random_seed(1337)


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