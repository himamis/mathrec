import numpy as np
from xainano_graphics import utils
from trainer.defaults import *


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', "<end>")
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


def xainano_sequence_generator(generator, config, parser, batch_size, vocabulary_map):
    while True:
        inputs = []
        input_sequences = []
        targets = []
        max_width = int(0)
        max_height = int(0)
        max_seq_len = int(0)
        for index in range(batch_size):
            tokens = []
            generator.generate_formula(tokens, config)
            image = parser.parse(tokens)
            input_sequence = list(tokens)
            input_sequence.insert(0, "<start>")
            tokens.append("<end>")

            inputs.append(image)
            input_sequences.append(np.array(input_sequence))
            targets.append(np.array(tokens))

            max_width = max(utils.w(image), max_width)
            max_height = max(utils.h(image), max_height)
            max_seq_len = max(max_seq_len, len(tokens))

        for index in range(batch_size):
            # Resize image
            image = inputs[index]
            rw, rh = max_width - utils.w(image), max_height - utils.h(image)
            left, top = int(rw / 2), int(rh / 2)
            inputs[index] = utils.pad_image(image, top, left, rh - top, rw - left)

            # Resize sequence
            seq = input_sequences[index]
            input_sequences[index] = np.pad(seq, (0, max_seq_len - len(seq)),  pad_with)
            input_sequences[index] = [vocabulary_map[token] for token in input_sequences[index]]

            t_seq = targets[index]
            targets[index] = np.pad(t_seq, (0, max_seq_len - len(t_seq)),  pad_with)
            targets[index] = [vocabulary_map[token] for token in targets[index]]

        yield [np.stack(inputs), np.reshape(np.stack(input_sequences), (batch_size, -1, 1))], \
               np.reshape(np.stack(targets), (batch_size, -1, 1))


def create_default_sequence_generator(token_parser, generator=create_generator(), config=create_config(), batch_size=1,
                                      vocabulary_map=create_vocabulary_map()):
    return xainano_sequence_generator(generator, config, token_parser, batch_size, vocabulary_map)
