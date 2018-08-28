import numpy as np
from xainano_graphics import utils
from trainer.defaults import *
from keras.utils import to_categorical


def xainano_sequence_generator(generator, config, parser, batch_size, vocabulary_map, height=256, width=512):
    vocabulary_size = len(vocabulary_map)
    while True:
        inputs = []
        input_sequences = []
        targets = []
        max_width = int(0)
        max_height = int(0)
        max_seq_len = int(0)
        for index in range(batch_size):
            while True:
                tokens = []
                generator.generate_formula(tokens, config)
                image = parser.parse(tokens)
                if utils.w(image) <= width and utils.h(image) <= height:
                    break

            input_sequence = list(tokens)
            input_sequence.insert(0, "<start>")
            tokens.append("<end>")

            inputs.append(image)
            input_sequences.append(np.array(input_sequence))
            targets.append(np.array(tokens))

            #max_width = max(utils.w(image), max_width)
            #max_height = max(utils.h(image), max_height)
            max_seq_len = max(max_seq_len, len(tokens))

        for index in range(batch_size):
            # Resize image
            image = inputs[index]
            rw, rh = width - utils.w(image), height - utils.h(image)
            left, top = int(rw / 2), int(rh / 2)
            inputs[index] = utils.pad_image(image, top, left, rh - top, rw - left)

            # Resize sequence
            seq = input_sequences[index]
            input_sequences[index] = np.append(seq, np.repeat('<end>', max_seq_len - len(seq)))
            input_sequences[index] = [vocabulary_map[token] for token in input_sequences[index]]

            t_seq = targets[index]
            targets[index] = np.append(t_seq, np.repeat('<end>', max_seq_len - len(t_seq)))
            targets[index] = [vocabulary_map[token] for token in targets[index]]

        yield [np.stack(inputs), to_categorical(input_sequences, vocabulary_size)], \
            to_categorical(targets, vocabulary_size)
        #yield [np.stack(inputs), np.array(input_sequences)], to_categorical(targets, vocabulary_size)


def create_default_sequence_generator(token_parser, generator=create_generator(), config=create_config(), batch_size=1,
                                      vocabulary_map=create_vocabulary_maps()):
    return xainano_sequence_generator(generator, config, token_parser, batch_size, vocabulary_map[0])
