import numpy as np
from graphics.utils import *


class Drawable(object):
    """ An object that can be drawn. """

    def draw(self) -> np.ndarray:
        raise NotImplementedError("Class %s doesn't implement draw()" % self.__class__.__name__)


class Graphics(Drawable):
    """ Graphics object that can draw stuff. """

    def token(self, token: str):
        raise NotImplementedError("Class %s doesn't implement token()" % self.__class__.__name__)

    def fraction(self, numerator: Drawable, denominator: Drawable):
        raise NotImplementedError("Class %s doesn't implement fraction()" % self.__class__.__name__)

    def square_root(self, expression: Drawable):
        raise NotImplementedError("Class %s doesn't implement square_root()" % self.__class__.__name__)

    def power(self, power: Drawable):
        raise NotImplementedError("Class %s doesn't implement power()" % self.__class__.__name__)


class DefaultGraphics(Graphics):

    def __init__(self):
        self._concatenator = ImageConcatenator()

    def draw(self) -> np.ndarray:
        return self._concatenator.draw()


class ImageConcatenator(Drawable):
    """ Utility class to help draw multiple images"""

    def __init__(self):
        self._images = []

    def append(self, image, y_center=None, margins=(0,0)):
        if y_center is None:
            y_center = round(h(image) / 2)
        self._images.append((image, y_center, margins))

    def draw(self) -> np.ndarray:
        width = 0
        height = 0
        paddings = []
        for image, y_center, margins in self._images:
            #padding = np.random.randint(0, 15)
            #paddings.append(padding)
            width += w(image)
            top_missing = abs(min(round(height / 2) - y_center, 0))
            bottom_missing = max(round(height / 2) + (h(image) - y_center) - height, 0)
            missing = max(top_missing, bottom_missing)
            height = height + missing * 2

        concat_image = new_image(width, height)

        offset = 0
        for i, (image, y_center, margins) in enumerate(self._images):
            paste(concat_image, image, offset, round(abs(h(concat_image) / 2 - y_center)))
            offset += w(image)# + paddings[i]
        return concat_image
