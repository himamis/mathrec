import numpy as np

from graphics import Drawable, Graphics
import inkml

graphics_backend = inkml.Graphics()


class InkmlGraphics(Graphics):

    def __init__(self, token_map):
        self._token_map = token_map

    def token(self, token: str):
        pass

    def fraction(self, numerator: Drawable, denominator: Drawable):
        pass

    def square_root(self, expression: Drawable):
        pass

    def power(self, power: Drawable):
        pass

    def draw(self) -> np.ndarray:
        pass
