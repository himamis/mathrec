from trainer import model, predictor
from trainer.defaults import *
import file_utils as utils
import numpy as np
from utilities import parse_arg
import cv2
from graphics import augment
from os import path
from trainer.sequence import create_default_sequence_generator
from xainano_graphics import postprocessor
from numpy.random import seed
from tensorflow import set_random_seed
import pickle


seed(1337)
set_random_seed(1337)



data_base_dir = parse_arg('--data-base-dir', '/Users/balazs/real_data')
weights_file = parse_arg('--weights', "/Users/balazs/university/models/model-att2-conv32-rowbilstm/weights_18.h5")
vocabulary = utils.read_pkl(path.join(data_base_dir, "vocabulary.pkl"))
vocabulary = vocabulary | {"<start>", "<end>", "^", "_", "\\frac", "{", "}", "\\mbox", "\\to", "\\left"} \
                        | {"\\right", "\\cdots"}
vocabulary = sorted(vocabulary)
vocabulary_maps = create_vocabulary_maps(vocabulary)


model, encoder, decoder = model.create_default(len(vocabulary))
if utils.file_exists(weights_file):
    print('Start loading weights')
    weights = utils.read_npy(weights_file)
    model.set_weights(weights)
    print('Weights loaded and set')
else:
    print("Weights file does not exist")
    exit()

predict = predictor.create_predictor(encoder, decoder, vocabulary, vocabulary_maps[0], vocabulary_maps[1])

images = pickle.load(open('/Users/balazs/real_data/data_training.pkl', 'rb'))

for image, truth in images:
    print(truth)
    prediction = predict(image)
    print(prediction)
    cv2.imshow(image, truth)
    cv2.waitKey(0)
