from graphics import utils
import cv2
import numpy as np
import pickle
import tensorflow as tf

_image_size = (224, 224)
_buffer_size = 100


def create_dataset_tensors(path, batch_size=32, shuffle=True, repeat=None):
    generator, dataset = create_generator(path)
    dataset = _create_dataset(generator, dataset.size(), batch_size=batch_size, shuffle=shuffle, repeat=repeat)
    return dataset


def create_generator(path):
    dataset = pickle.load(open(path, 'rb'))
    data_generator = DataGenerator(dataset, image_size=_image_size, batch_size=32)

    def gen():
        yield data_generator.next()
    return gen, data_generator


def _create_dataset(generator, length, batch_size=32, shuffle=True, repeat=None):
    dataset = tf.data.Dataset.from_generator(
        generator,
        (tf.int32, tf.int32),
        (_image_size + (1,), ())
    ).take(length)
    if repeat is not None:
        dataset = dataset.repeat(repeat)
    if shuffle:
        dataset = dataset.shuffle(_buffer_size)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    return dataset


def pad(missing):
    side_1 = int(missing / 2)
    side_2 = side_1
    if missing % 2 == 1:
        side_2 += 1

    return side_1, side_2


class DataGenerator(object):

    def __init__(self, dataset, image_size, batch_size=None, do_shuffle=True, steps=None):
        _, self.images, self.labels = zip(*dataset)
        self.images = list(self.images)
        self.labels = list(self.labels)
        self.batch_size = batch_size
        self.image_size = image_size

        self._pointer = 0
        self._steps = steps
        self.do_shuffle = do_shuffle
        self.classes = set()
        self._read_classes()
        self._encoder = { item: index + 1 for index, item in enumerate(sorted(self.classes)) }
        self.encoded_labels = [self._encoder[label] for label in self.labels]
        self._transform()
        self._assert()

    def _read_classes(self):
        for label in self.labels:
            self.classes.add(label)

    def _assert(self):
        for index, image in enumerate(self.images):
            assert utils.w(image) == self.image_size[0] and utils.h(image) == self.image_size[1], \
            "Problem with index {}".format(index)
            assert image.shape == (self.image_size[0], self.image_size[1], 1), index

    def _transform(self):
        for index, image in enumerate(self.images):
            aspect_ratio = float(utils.w(image)) / float(utils.h(image))

            if utils.w(image) > self.image_size[0]:
                image = utils.resize(image, self.image_size[0], int(round(self.image_size[0] / aspect_ratio)))
            if utils.h(image) > self.image_size[1]:
                image = utils.resize(image, int(round(self.image_size[1] * aspect_ratio)), self.image_size[1])

            if utils.w(image) < self.image_size[0] or utils.h(image) < self.image_size[1]:
                missing_w = self.image_size[0] - utils.w(image)
                missing_h = self.image_size[1] - utils.h(image)
                top, bottom = pad(missing_h)
                left, right = pad(missing_w)
                image = utils.pad_image(image, top, left, bottom, right, 0, 1, cv2.bitwise_or)

            image = np.reshape(image, self.image_size + (1,))

            self.images[index] = image

    def size(self):
        return len(self.images)

    def next(self):
        image = self.images[self._pointer]
        label = self.encoded_labels[self._pointer]

        self._pointer += 1
        if self._pointer == len(self.images):
            self._pointer = 0

        return image, label
