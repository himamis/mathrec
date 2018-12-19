from graphics import augment
from utilities import parse_arg
from file_utils import read_img
import cv2
from numpy import random
from trainer.defaults import *
from xainano_graphics import postprocessor

random.seed(123)

data_base_dir = parse_arg("--data-base-dir", "/Users/balazs/university/xainano_images")
background_images_dir = parse_arg("--background-images-dir", "/Volumes/SDCard/downloads/paper texture background")
grid_images_dir = parse_arg("--grid-images-dir", "/Volumes/SDCard/downloads/graph paper printable")

generator = create_generator()
config = create_config()
vocabulary = create_vocabulary(generator, config)
vocabulary_maps = create_vocabulary_maps(vocabulary)
token_parser = create_token_parser(data_base_dir)

# generate data generators
augmentor = augment.Augmentor(background_images_dir, grid_images_dir)
post_processor = postprocessor.Postprocessor()

image = "/Users/balazs/university/generated_images10/TC11_package/images/CROHME2016_data_TEST2016_INKML_GT_UN_120_em_422.png"
image = read_img(image)

while True:
    tokens = []
    generator.generate_formula(tokens, config)
    image = token_parser.parse(tokens, post_processor)
    #image = augmentor.size_changing_augment(image)
    #image = augmentor.augment(image)

    cv2.imshow("Image", image)
    cv2.waitKey(0)
