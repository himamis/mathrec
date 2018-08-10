import graphics as g
import cv2
import numpy as np


def _should_minimize(token):
    return token == "*" or token == "+" or token == "-" or token == "=" or token == "times" or \
           token == "geq" or token == "leq" or token == "neq"


def _minimize(image):
    # TODO randomize new size
    return g._resize(image, 30, 30)


class Preprocessor:

    def preprocess(self, image, token):
        if _should_minimize(token):
            image = _minimize(image)
        if token == ",":
            rows, cols, ch = image.shape
            transform = np.float32([[1, 0, 0], [0, 1, 30]])
            image = cv2.warpAffine(image, transform, (rows, cols), borderValue=(255, 255, 255))
            new_width = 20
            image = g._subimage(image, round((g._w(image) - new_width) / 2), 0, 20, g._h(image))

        return image
