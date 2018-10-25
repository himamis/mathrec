from graphics import augment
from utilities import parse_arg
from file_utils import read_img
import cv2
from numpy import random

random.seed(123)

base_dir = parse_arg("--data-base-dir", "/Users/balazs/university/mathrec/backgrounds")

image = "/Users/balazs/university/generated_images10/TC11_package/images/CROHME2016_data_TEST2016_INKML_GT_UN_120_em_422.png"
image = read_img(image)
augmentor = augment.Augmentor(base_dir)

while True:
    new_image = augmentor.rotate(image)
    new_image = augmentor.background(new_image)
    new_image = augmentor.blur(new_image)

    cv2.imshow("Image", new_image)
    cv2.waitKey(0)
