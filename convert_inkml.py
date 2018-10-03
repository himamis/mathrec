from inkml import Graphics
from inkml import InkML
import cv2
import numpy as np


if __name__ == "__main__":
    file = open("/Users/balazs/university/TC11_package/CROHME2016_data/TEST2016_INKML_GT/UN_101_em_4.inkml")
    string = file.read()
    inkml = InkML(string)

    graphics = Graphics()
    graphics.start_image(inkml)
    image = graphics.end_image()

    pts = [np.reshape(np.array(line), (-1, 1, 2)) for line in inkml.symbols]

    cv2.polylines(image, pts, False, (0, 0, 0), 20)
    #for symbol in inkml.symbols:
    #    if len(symbol) > 1:
    #        graphics.new_symbol()
    #        for i in range(0, len(symbol)):
    #            graphics.point(symbol[i])
    #    elif len(symbol) == 1:
    #        print("Symbol consists of a single point. Ignoring.")
    #    else:
    #        print("Symbol is missing points.")
    #image = graphics.end_image()
    cv2.imshow('image', image)
    a = 1