from graphics import utils
from sklearn.utils import shuffle
import numpy as np
import cv2



class DataGenerator:

    # levels is the number of difficulty levels

    def __init__(self,
                 images,
                 labels,
                 encoding_vb,
                 batch_size=32,
                 ):
        self.images = images
        self.labels = labels
        self.encoding_vb = encoding_vb
        self.batch_size = batch_size
        self.chunk_index = 0
        self.image_chuncks = None
        self.label_chuncks = None
        self._build_chunks()

    def _build_chunks(self):
        lengs = [len(lab) for lab in self.labels]
        _, labels, images = zip(*sorted(zip(lengs, self.labels, self.images), key=lambda a: a[0]))
        self.label_chuncks = [labels[i:i + self.batch_size] for i in range(0, len(labels), self.batch_size)]
        self.image_chuncks = [images[i:i + self.batch_size] for i in range(0, len(images), self.batch_size)]

        if len(self.image_chuncks[-1]) != self.batch_size:
            del self.image_chuncks[-1]
            del self.label_chuncks[-1]

    def reset(self):
        self.image_chuncks, self.label_chuncks = shuffle(self.image_chuncks, self.label_chuncks)
        self.chunk_index = 0

    def steps(self):
        return len(self.image_chuncks)

    def next_batch(self):
        image_bucket = self.image_chuncks[self.chunk_index]
        label_bucket = self.label_chuncks[self.chunk_index]
        self.chunk_index += 1

        images = []
        labels = []
        observations = []
        max_image_width = -1
        max_image_height = -1
        max_label_length = -1

        for i in range(len(image_bucket)):
            image = image_bucket[i]
            label = label_bucket[i]

            images.append(image)
            labels.append(label)
            observations.append(label)

            max_image_width = max(utils.w(image), max_image_width)
            max_image_height = max(utils.h(image), max_image_height)
            max_label_length = max(len(label) + 1, max_label_length)

        lengths = [len(label) + 1 for label in labels]  # + 1 for end/start label
        image_masks = []

        for i in range(len(image_bucket)):
            # Resize image
            image = images[i]

            h, w = utils.h(image), utils.w(image)
            rw, rh = max_image_width - w, max_image_height - h
            left, top = int(rw / 2), int(rh / 2)
            image = utils.pad_image(image, top, left, rh - top, rw - left, 0, 1)
            images[i] = image

            image_mask = np.zeros((max_image_height, max_image_width, 1), dtype=np.float32)
            image_mask[top:(top + h), left:(left+w),:] = 1.0
            image_masks.append(image_mask)


        # create labels and observations
        start_id = self.encoding_vb['<start>']
        end_id = self.encoding_vb['<end>']
        labels = [[self.encoding_vb[char] for char in label] for label in labels]
        observations = list(labels)

        masks = [np.append(np.append(np.ones(len(label), dtype=np.float32),
                                     max(min((((len(label) + 1) * (len(label) + 1)) / (len(self.encoding_vb)-1)), 1), 0.1)
                                     ), np.zeros(max_label_length - len(label) - 1, dtype=np.float32)) for label in labels]
        labels = [label + [end_id] * (max_label_length - len(label)) for label in labels]
        observations = [[start_id] + label + [end_id] * (max_label_length - len(label) - 1) for label in observations]

        return images, labels, observations, image_masks, masks


class DifficultyDataGenerator:

    def __init__(self,
                 images,
                 labels,
                 encoding_vb,
                 levels=3,
                 batch_size=32):
        self.images = images
        self.labels = labels
        self.encoding_vb = encoding_vb
        self.levels = levels
        self.batch_size = batch_size
        self.chunk_index = 0
        self.image_chuncks = None
        self.label_chuncks = None
        self.difficulties = []
        self._build_chunks()

    def _build_chunks(self):
        lengs = [len(lab) for lab in self.labels]
        lengs, labels, images = zip(*sorted(zip(lengs, self.labels, self.images), key=lambda a: a[0]))

        pcts = np.array([(i + 1) / self.levels for i in range(self.levels)])
        ids = np.floor(len(lengs) * pcts)

        for id in ids:
            cur_id = int(id)
            d_labels = labels[:cur_id]
            d_images = images[:cur_id]

            d_label_chuncks = [d_labels[i:i + self.batch_size] for i in range(0, len(d_labels), self.batch_size)]
            d_images_chuncks = [d_images[i:i + self.batch_size] for i in range(0, len(d_images), self.batch_size)]

            # Remove last batch if it doesn't match size
            if len(d_label_chuncks[-1]) != self.batch_size:
                del d_label_chuncks[-1]
                del d_images_chuncks[-1]

            self.difficulties.append((d_label_chuncks, d_images_chuncks))

        self.set_level(0)

    def set_level(self, level):
        self.label_chuncks, self.image_chuncks = self.difficulties[level]


    def reset(self):
        self.image_chuncks, self.label_chuncks = shuffle(self.image_chuncks, self.label_chuncks)
        self.chunk_index = 0

    def steps(self):
        #return len(self.difficulties[-1][0])
        return len(self.image_chuncks)

    def next_batch(self):
        if self.chunk_index == len(self.image_chuncks):
            self.chunk_index = 0
        image_bucket = self.image_chuncks[self.chunk_index]
        label_bucket = self.label_chuncks[self.chunk_index]
        self.chunk_index += 1

        images = []
        labels = []
        observations = []
        max_image_width = -1
        max_image_height = -1
        max_label_length = -1

        for i in range(len(image_bucket)):
            image = image_bucket[i]
            label = label_bucket[i]

            images.append(image)
            labels.append(label)
            observations.append(label)

            max_image_width = max(utils.w(image), max_image_width)
            max_image_height = max(utils.h(image), max_image_height)
            max_label_length = max(len(label) + 1, max_label_length)

        lengths = [len(label) + 1 for label in labels]  # + 1 for end/start label
        image_masks = []

        for i in range(len(image_bucket)):
            # Resize image
            image = images[i]

            h, w = utils.h(image), utils.w(image)
            rw, rh = max_image_width - w, max_image_height - h
            left, top = int(rw / 2), int(rh / 2)
            image = utils.pad_image(image, top, left, rh - top, rw - left, 0, 1, operation=cv2.bitwise_or)
            images[i] = image

            image_mask = np.zeros((max_image_height, max_image_width, 1), dtype=np.float32)
            image_mask[top:(top + h), left:(left+w),:] = 1.0
            image_masks.append(image_mask)


        # create labels and observations
        start_id = self.encoding_vb['<start>']
        end_id = self.encoding_vb['<end>']
        labels = [[self.encoding_vb[char] for char in label] for label in labels]
        observations = list(labels)

        masks = [np.append(np.append(np.ones(len(label), dtype=np.float32), 1 / len(self.encoding_vb)), np.zeros(max_label_length - len(label) - 1, dtype=np.float32)) for label in labels]
        labels = [label + [end_id] * (max_label_length - len(label)) for label in labels]
        observations = [[start_id] + label + [end_id] * (max_label_length - len(label) - 1) for label in observations]

        return images, labels, observations, image_masks, masks
