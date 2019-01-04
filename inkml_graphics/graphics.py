from graphics.utils import *
from graphics import Drawable, DefaultGraphics
import inkml

graphics_backend = inkml.Graphics()
high_characters = ['b', 'd', 'h', 'k', 'l', 't']
low_characters = ['g', 'j', 'p', 'q', 'y']


class InkmlGraphics(DefaultGraphics):

    def __init__(self, token_map):
        super().__init__()
        self._token_map = token_map

    def _random_trace(self, token):
        traces = self._token_map[token]
        return np.random.choice(traces)

    def _random_image(self, token, expected_width=None, expected_height=None, padding=6):
        trace = self._random_trace(token)
        return graphics_backend.create_token_image(trace,
                                                   expected_width=expected_width,
                                                   expected_height=expected_height,
                                                   padding=padding)

    def token(self, token: str):
        if token in high_characters:
            image = self._random_image(token, expected_height=65)
            self._concatenator.append(image, 45)
        elif token in low_characters:
            image = self._random_image(token, expected_height=65)
            self._concatenator.append(image, 25)
        elif token == '.':
            image = self._random_image(token, expected_width=5, expected_height=5, padding=0)
            self._concatenator.append(image, -8)
        else:
            if token == '-':
                image = self._random_image(token, expected_width=35)
            elif token == '.':
                image = self._random_image(token, expected_width=7, expected_height=15, padding=2)
            elif token == '+':
                image = self._random_image(token, expected_width=35)
            else:
                image = self._random_image(token, expected_height=50)
            self._concatenator.append(image)

    def fraction(self, numerator: Drawable, denominator: Drawable):
        num_im = numerator.draw()
        denom_im = denominator.draw()
        width = max(w(num_im), w(denom_im)) + np.random.randint(-5, 10)
        frac_trace = self._random_trace("-")
        frac_im = graphics_backend.create_token_image(frac_trace, expected_width=width, padding=4)
        height = h(num_im) + h(denom_im) + h(frac_im)
        width = max((w(num_im), w(denom_im), w(frac_im)))
        new_im = new_image(width, height)
        paste(new_im, num_im, 0, 0)
        paste(new_im, frac_im, 0, h(num_im))
        paste(new_im, denom_im, 0, h(num_im) + h(frac_im))
        self._concatenator.append(new_im, round(h(num_im) + h(frac_im) / 2))

    def square_root(self, expression: Drawable):
        pass

    def power(self, power: Drawable):
        pass

