import cv2
import numpy as np


def _w(image):
    return image.shape[1]


def _h(image):
    return image.shape[0]


def _center(image1, image2, dim):
    return round(abs(dim(image1) - dim(image2)) / 2)


def _paste(destination, source, x, y):
    subarray = destination[y: y + source.shape[0], x: x + source.shape[1]]
    cv2.bitwise_and(subarray, source, subarray)


def _insert(destination, source, x, y):
    destination[y: y + source.shape[0], x: x + source.shape[1]] = source


def _new_image(width, height, initial_value=255):
    if initial_value is None:
        return np.ndarray((height, width, 3), dtype=np.uint8)
    else:
        return np.full((height, width, 3), initial_value, dtype=np.uint8)


def _resize(image, new_width, new_height):
    interp = cv2.INTER_LINEAR
    if new_width < _w(image) or new_height < _h(image):
        interp = cv2.INTER_AREA
    return cv2.resize(image, (new_width, new_height), interpolation=interp)


def _subimage(image, x, y, width, height):
    return image[y: y + height, x: x + width]


def _pad_image(image, top, left, bottom, right, default_value = 255):
    new_image = _new_image(_w(image) + left + right, _h(image) + top + bottom, default_value)
    _paste(new_image, image, left, top)
    return new_image


def _concat(images):
    width = 0
    height = 0
    for image in images:
        width += _w(image)
        height = max(height, _h(image))

    new_image = _new_image(width, height)

    offset = 0
    for image in images:
        _paste(new_image, image, offset, _center(new_image, image, _h))
        offset += _w(image)
    return new_image


class Graphics:

    def __init__(self):
        self.images = []

    def expression(self, expression):
        if isinstance(expression, list):
            self.images += expression
        else:
            self.images.append(expression)

    def fraction(self, numerator, denominator, fraction_line):
        width = max(_w(numerator), _w(denominator))
        # TODO: add randomness in width
        fraction_line = _resize(fraction_line, width, _h(fraction_line))

        fraction = _new_image(width, _h(numerator) + _h(denominator) + _h(fraction_line))
        offset = 0
        _paste(fraction, numerator, round((width - _w(numerator)) / 2), offset)
        offset += _h(numerator)
        _paste(fraction, fraction_line, 0, offset)
        offset += _h(fraction_line)
        _paste(fraction, denominator, 0, offset)

        self.images.append(fraction)

    def square_root(self, square_root, expression):
        stretch_start = 20
        stretch_end = 30
        square_root = _resize(square_root, _w(square_root), _h(expression) + 40)
        pre = _subimage(square_root, 0, 0, stretch_start, _h(square_root))
        center = _subimage(square_root, stretch_start, 0, stretch_end - stretch_start, _h(square_root))
        post = _subimage(square_root, stretch_end, 0, _w(square_root), _h(square_root))
        center = _resize(center, _w(expression), _h(square_root))
        new_square_root = _concat([pre, center, post])
        _paste(new_square_root, expression, stretch_start, 30)

        self.images.append(new_square_root)

    def power(self, power):
        small_power = _resize(power, round(_w(power) / 2), round(_h(power) / 2))
        new_image = _pad_image(small_power, 0, 0, round(_h(power) / 2) + 30, 0)
        # TODO vary the positions

        self.images.append(new_image)

    def draw(self):
        return _concat(self.images)
