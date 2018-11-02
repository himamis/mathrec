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


data_base_dir = parse_arg('--data-base-dir', '/Users/balazs/university/xainano_images')
weights_file = parse_arg('--weights', "/Users/balazs/university/weights_9.h5")
background_dir = '/Volumes/SDCard/split_backgrounds_dir'

generator = create_generator()
token_parser = create_token_parser(data_base_dir)
config = create_config()
vocabulary = create_vocabulary(generator, config)
encoding_vb, decoding_vb = create_vocabulary_maps(vocabulary)
train_augmentor = augment.Augmentor(path.join(background_dir, 'training/backgrounds'), path.join(background_dir, 'training/grids'))
post_processor = postprocessor.Postprocessor()

data = create_default_sequence_generator(token_parser, train_augmentor, post_processor, generator, config, 1, [encoding_vb, decoding_vb])

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

predict = predictor.create_predictor(encoder, decoder, vocabulary, encoding_vb, decoding_vb)
custom_images = False

while True:
    if custom_images:
        print("Image path: \n")
        input_image = input()
        image = utils.read_img(input_image)
        image = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 2)
    else:
        print("Generated images")
        image = next(data)[0][0][0]

    cv2.imshow("Image", image)
    prediction = predict(image)
    print("Prediction: " + prediction + "\n")
    cv2.waitKey(0)
