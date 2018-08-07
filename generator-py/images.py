import os
import numpy as np
import cv2


base = "/Users/balazs/university/extracted_images"


def _token2image(token):
    if token.startswith("\\"):
        return token[1:]
    elif token == "<":
        return "geq"
    else:
        return token


def _dot():



class Images:

    def __init__(self):
        self.directories = next(os.walk(base))[1]
        self.images = {}
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
        else:
            raise Exception("Unknown token: " + token)

    def _random_image(self, token):
        count = len(self.images[token])
        index = np.random.randint(count)
        path = os.path.join(base, token, self.images[token][index])

        return cv2.imread(path)
