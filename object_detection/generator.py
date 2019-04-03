from graphics import utils
import cv2
import numpy as np
import pickle
import tensorflow as tf
from sklearn.utils import class_weight

_image_size = (224, 224)
_buffer_size = 1000


def create_dataset_tensors(path, batch_size=32, shuffle=True, repeat=None):
    data_generator = _create_data_generator(path)

    def data_generator_function():
        for i in range(data_generator.size()):
            yield data_generator.next()

    dataset = _create_dataset(data_generator_function, batch_size=batch_size, shuffle=shuffle, repeat=repeat)
    return dataset


def _create_data_generator(path):
    dataset = pickle.load(open(path, 'rb'))
    data_generator = DataGenerator(dataset, image_size=_image_size, batch_size=32)

    return data_generator


def _create_dataset(generator, batch_size=32, shuffle=True, repeat=None):
    dataset = tf.data.Dataset.from_generator(
        generator,
        (tf.int32, tf.int32),
        (_image_size + (1,), ())
    )
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

        cls_weights = class_weight.compute_class_weight('balanced', np.unique(self.labels), self.labels)
        print(cls_weights)

        number = {lab: 0 for lab in self.labels}
        for _, image, label in dataset:
            number[label] += 1

        number = {self._encoder[label]: value for label, value in number.items()}
        weights = [number[key] for key in sorted(number.keys())]
        weights = [weight / sum(weights) for weight in weights]
        print(weights)


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
            # aspect_ratio = float(utils.w(image)) / float(utils.h(image))
            image = utils.resize(image, _image_size[0], _image_size[1])

            # if utils.w(image) > self.image_size[0]:
            #     image = utils.resize(image, self.image_size[0], int(round(self.image_size[0] / aspect_ratio)))
            # if utils.h(image) > self.image_size[1]:
            #     image = utils.resize(image, int(round(self.image_size[1] * aspect_ratio)), self.image_size[1])
            #
            # if utils.w(image) < self.image_size[0] or utils.h(image) < self.image_size[1]:
            #     missing_w = self.image_size[0] - utils.w(image)
            #     missing_h = self.image_size[1] - utils.h(image)
            #     top, bottom = pad(missing_h)
            #     left, right = pad(missing_w)
            #     image = utils.pad_image(image, top, left, bottom, right, 0, 1, cv2.bitwise_or)

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


empirical_class_weights = [8.61026899150092, 0.20661720026324146, 0.20683681888597022, 0.16050154999003693, 1.2522012819042523, 0.10855298404993907, 1.2990125447791776, 4.070964000165707, 0.46821963235784597, 0.13869713414677176, 0.13806731886470894, 0.357311933910982, 0.53400680353436, 0.8710478030793232, 1.0932139281343864, 1.1460075336155524, 1.2071468933493845, 1.1865370683409804, 0.2334357955949146, 2.8283732443011744, 3.4259168874633943, 3.1900340853757507, 4.964083653263286, 4.889248221304543, 8.534740316136876, 7.661105480626802, 10.135004125412541, 5.259245383997859, 6.9497171145686, 5.624048532020832, 5.896729672967297, 4.122713542540695, 5.861207205057855, 6.486402640264027, 6.157977190124075, 2.820175060984359, 7.484310738766184, 17.3742927864215, 4.157950410425658, 4.157950410425658, 2.014410757845971, 2.8366192304361633, 1.401960224840928, 5.896729672967297, 108.1067106710671, 46.33144743045733, 9.35538842345773, 5.656746488602349, 13.703667549853577, 32.43201320132013, 2.350145884153633, 1.471952187654469, 31.385819227084, 5.826110155925773, 3.8456932649786717, 2.7484756950271296, 3.0405012376237623, 7.484310738766184, 15.443815810152444, 7.370912091209121, 9.72960396039604, 1.7594220543211645, 5.624048532020832, 7.661105480626802, 2.643914119672837, 14.741824182418242, 1.0552715792186593, 0.5043858973766739, 1.430824111822947, 3.3093891021755235, 1.6717532578000067, 1.4543503677722032, 11.183452828041425, 11.183452828041425, 0.35316166825393974, 0.5518777062051072, 0.9364392647156919, 0.7916683450281562, 1.6216006600660067, 1.2818977549928905, 2.7407335099707155, 3.3435065156000134, 0.8877375876273759, 2.9483648364836483, 1.3024904900128567, 7.101900701019007, 1.8323171300180865, 0.3812540736832304, 7.484310738766184, 1.5108080683844782, 2.803920449681856, 1.8118443129229125, 3.3435065156000134, 1.1610505919327017, 2.540366569294005, 2.795863207010356, 5.861207205057855, 0.169122265955085, 0.4923888643925121, 0.8148747035507571, 1.815224619476873]