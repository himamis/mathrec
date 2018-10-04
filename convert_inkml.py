from inkml import Graphics
from inkml import InkML
from utilities import parse_arg
import cv2

data_base_dir = parse_arg('--data-base-dir', '/Users/balazs/university/')
output_dir = parse_arg('--output-dir', '/Users/balazs/university/handwritten_images')




if __name__ == "__main__":
    file = open("/Users/balazs/university/TC11_package/CROHME2016_data/TEST2016_INKML_GT/UN_101_em_4.inkml")
    string = file.read()
    inkml = InkML(string)

    graphics = Graphics()
    image = graphics.create_image(inkml)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    a = 1