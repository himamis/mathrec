import cv2
import numpy as np


def w(image):
    return image.shape[1]


def h(image):
    return image.shape[0]


def center(image1, image2, dim):
    return round(abs(dim(image1) - dim(image2)) / 2)


def paste(destination, source, x, y):
    subarray = destination[y: y + source.shape[0], x: x + source.shape[1]]
    cv2.bitwise_or(subarray, source, subarray)


def insert(destination, source, x, y):
    destination[y: y + source.shape[0], x: x + source.shape[1]] = source


def new_image(width, height, initial_value=255, channels=3):
    if initial_value is None:
        return np.ndarray((height, width, channels), dtype=np.uint8)
    else:
        return np.full((height, width, channels), initial_value, dtype=np.uint8)


def resize(image, new_width, new_height):
    interp = cv2.INTER_LINEAR
    if new_width < w(image) or new_height < h(image):
        interp = cv2.INTER_AREA
    return cv2.resize(image, (new_width, new_height), interpolation=interp)


def sub_image(image, x, y, width, height):
    return image[y: y + height, x: x + width]


def pad_image(image, top, left, bottom, right, default_value = 255, channels=3):
    padded_image = new_image(w(image) + left + right, h(image) + top + bottom, default_value, channels)
    paste(padded_image, image, left, top)
    return padded_image


def concat(images):
    width = 0
    height = 0
    for image in images:
        width += w(image)
        height = max(height, h(image))

    concat_image = new_image(width, height)

    offset = 0
    for image in images:
        paste(concat_image, image, offset, center(concat_image, image, h))
        offset += w(image)
    return concat_image
