import os
import numpy as np
import cv2
from graphics.utils import new_image
import file_utils
import logging

def _token2image(token):
    if token == "\\frac":
        return "-"
    elif token.startswith("\\"):
        return token[1:]
    elif token == "<":
        return "geq"
    else:
        return token


def _dot():
    dot = new_image(20, 45, 255)
    cv2.circle(dot, (10, 22), 10, (255, 255, 255, 0))
    return dot

_images = ['!', '1', '8', 'H', '[', 'd', 'gamma', 'j', 'log', 'pi', 'sqrt', 'w', '(', '2', '9', 'M', ']', 'div', 'geq',
           'k', 'lt', 'pm', 'sum', 'y', ')', '3', '=', 'N', 'alpha', 'e', 'gt', 'l', 'mu', 'prime', 'tan', 'z', '+',
           '4', 'A', 'R', 'ascii_124', 'exists', 'i', 'lambda', 'neq', 'q', 'theta', '{', ',', '5', 'C', 'S', 'b', 'f',
           'in', 'ldots', 'o', 'rightarrow', 'times', '}', '-', '6', 'Delta', 'T', 'beta', 'forall', 'infty', 'leq', 'p',
           'sigma', 'u', '0', '7', 'G', 'X', 'cos', 'forward_slash', 'int', 'lim', 'phi', 'sin', 'v']

class Images:

    def __init__(self, base, preprocessor):
        self.base = base
        self.directories = _images
        self.images = {}
        self.preprocessor = preprocessor
        logging.info("Loading directories: " + str(self.directories))
        for directory in self.directories:
            path = os.path.join(self.base, directory)
            self.images[directory] = file_utils.list_files(path)
            logging.info("Loaded directory: " + directory)

    def image(self, token):
        token = _token2image(token)
        if token in self.images:
            return self._random_image(token)
        elif token.upper() in self.images:
            return self._random_image(token.upper())
        elif token.lower() in self.images:
            return self._random_image(token.lower())
        elif token == ".":
            return _dot()
        else:
            raise Exception("Unknown token: " + token)

    def _random_image(self, token):
        count = len(self.images[token])
        index = np.random.randint(count)
        path = self.images[token][index]
        image = file_utils.read_img(path)#cv2.imread(path)
        image = self.preprocessor.preprocess(image, token)

        return image
