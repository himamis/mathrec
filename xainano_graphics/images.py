import os
import numpy as np
import cv2
from .utils import new_image


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


class Images:

    def __init__(self, base, preprocessor):
        self.directories = next(os.walk(base))[1]
        self.images = {}
        self.base = base
        self.preprocessor = preprocessor
        for directory in self.directories:
            self.images[directory] = next(os.walk(os.path.join(base, directory)))[2]

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
        path = os.path.join(self.base, token, self.images[token][index])
        image = cv2.imread(path)
        image = self.preprocessor.preprocess(image, token)

        return image
