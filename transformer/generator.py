import numpy as np
from trainer import tf_generator
from transformer import vocabulary

class DataGenerator(object):

    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.chunk_index = 0
        self.data_chunks = None
        self.formula_chunck = None
        self._build_chunks()

    def _build_chunks(self):
        lengs = [len(inputs) for formula, inputs in self.data]
        _, data = zip(*sorted(zip(lengs, self.data), key=lambda a: a[0]))
        self.data_chuncks = [data[i:i + self.batch_size] for i in range(0, len(data), self.batch_size)]

        if len(self.data_chuncks[-1]) == 1 and self.batch_size != 1:
            del self.data_chuncks[-1]
        self.reset()

    def reset(self):
        np.random.shuffle(self.data_chuncks)
        self.chunk_index = 0

    def steps(self):
        return len(self.data_chuncks)

    def next_batch(self):
        data_bucket = self.data_chuncks[self.chunk_index]
        self.chunk_index += 1
        print(self.chunk_index)

        formulas, inputs = zip(*data_bucket)
        tokens = []
        bounding_boxes = []
        for input in inputs:
            token, bounding_box = zip(*input)
            tokens.append(list(token))
            bounding_boxes.append(list(bounding_box))
        #tokens, bounding_boxes = zip(*inputs)

        encoded_formulas = tf_generator.encode_sequences(formulas, vocabulary.encoding_vocabulary)
        encoded_formulas, encoded_formulas_masks = tf_generator.normalize_sequences(encoded_formulas, vocabulary.EOS_ID)

        encoded_tokens = tf_generator.encode_sequences(tokens, vocabulary.encoding_vocabulary)
        encoded_tokens, encoded_tokens_masks = tf_generator.normalize_sequences(encoded_tokens, vocabulary.EOS_ID)

        num = len(max(encoded_tokens, key=len))
        bounding_boxes = [np.concatenate((np.array(box), np.repeat(0, 4 * (num - len(box))).reshape((-1, 4))))
                          for box in bounding_boxes]

        return encoded_tokens, bounding_boxes, encoded_formulas, encoded_formulas_masks
