from graphics import utils
from sklearn.utils import shuffle


class DataGenerator:

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
        self.data_index = 0

    def next_batch(self):
        images = []
        labels = []
        observations = []
        max_image_width = -1
        max_image_height = -1
        max_label_length = -1

        for i in range(self.batch_size):
            image = self.images[self.data_index]
            label = self.labels[self.data_index]

            images.append(image)
            labels.append(label)
            observations.append(label)

            max_image_width = max(utils.w(image), max_image_width)
            max_image_height = max(utils.h(image), max_image_height)
            max_label_length = max(len(label) + 1, max_label_length)

            self.data_index += 1

        lengths = [len(label) + 1 for label in labels]  # + 1 for end/start label

        for i in range(self.batch_size):
            # Resize image
            image = images[i]

            rw, rh = max_image_width - utils.w(image), max_image_height - utils.h(image)
            left, top = int(rw / 2), int(rh / 2)
            image = utils.pad_image(image, top, left, rh - top, rw - left, 0, 1)
            images[i] = image

        # create labels and observations
        start_id = self.encoding_vb['<start>']
        end_id = self.encoding_vb['<end>']
        labels = [[self.encoding_vb[char] for char in label] for label in labels]
        observations = list(labels)

        labels = [label + [end_id] * (max_label_length - len(label)) for label in labels]
        observations = [[start_id] + label + [end_id] * (max_label_length - len(label) - 1) for label in observations]

        return images, labels, observations, lengths

    def reset(self):
        self.images, self.labels = shuffle(self.images, self.labels)
        self.data_index = 0
