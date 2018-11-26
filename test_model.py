import file_utils as utils
from trainer import model
from utilities import parse_arg
from numpy.random import seed
from os import path
from graphics import augment

from tensorflow import set_random_seed
from trainer.sequence import predefined_image_sequence_generator
from trainer.defaults import *
from trainer.predictor import create_predictor

seed(1337)
set_random_seed(1337)

weights_fname = parse_arg('--weights', 'weights_20.h5')
data_base_dir = parse_arg('--data-base-dir', '/Users/balazs/university/model')

max_length = 200

vocabulary = utils.read_pkl(path.join(data_base_dir, "vocabulary.pkl"))
vocabulary = vocabulary | {"<start>", "<end>", "^", "_", "\\frac", "{", "}", "\\mbox", "\\to", "\\left"} \
                        | {"\\right", "\\cdots"}
vocabulary = sorted(vocabulary)
vocabulary_maps = create_vocabulary_maps(vocabulary)

model, encoder, decoder = model.create_default(len(vocabulary), None)

if not utils.file_exists(weights_fname):
    print("weights file does not exist: " + weights_fname)
    exit(1)

weights = utils.read_npy(weights_fname)
model.set_weights(weights)


images = utils.read_pkl(path.join(data_base_dir, "data_test_2014.pkl"))
#x_test, y_test = zip(*images)

#test_data = predefined_image_sequence_generator(x_test, y_test, vocabulary_maps[0], None, 1)

predict = create_predictor(encoder, decoder, vocabulary, vocabulary_maps[0], vocabulary_maps[1], 100)
augmentor = augment.Augmentor()

for image, truth in images:
    grayscale_image = augmentor.grayscale(image)

    predicted = predict(grayscale_image)

    print(truth)
    print(predicted)

