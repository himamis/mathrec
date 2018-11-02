from graphics import utils
from trainer.defaults import *
from keras.utils import to_categorical
from file_utils import *


def xainano_sequence_generator(generator, config, parser, batch_size, vocabulary_map, augmentor, post_processor):
    vocabulary_size = len(vocabulary_map)
    while True:
        inputs = []
        input_sequences = []
        targets = []
        max_width = int(0)
        max_height = int(0)
        max_seq_len = int(0)
        for index in range(batch_size):
            image = None
            tries = 0
            while image is None and tries < 10:
                try:
                    tries += 1
                    tokens = []
                    generator.generate_formula(tokens, config)
                    image = parser.parse(tokens, post_processor)
                    image = augmentor.size_changing_augment(image)
                except:
                    image = None
                    print("Unexpected error while parsing image, retry no: " + str(tries))
            
            if image is None:
                raise ValueError
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
            padded_image = utils.pad_image(image, top, left, rh - top, rw - left)
            inputs[index] = augmentor.augment(padded_image)

            # Resize sequence
            seq = input_sequences[index]
            input_sequences[index] = np.append(seq, np.repeat('<end>', max_seq_len - len(seq)))
            input_sequences[index] = [vocabulary_map[token] for token in input_sequences[index]]

            t_seq = targets[index]
            targets[index] = np.append(t_seq, np.repeat('<end>', max_seq_len - len(t_seq)))
            targets[index] = [vocabulary_map[token] for token in targets[index]]

        yield [np.stack(inputs), to_categorical(input_sequences, vocabulary_size)], \
            to_categorical(targets, vocabulary_size)


def create_default_sequence_generator(token_parser, augmentor, post_processor, generator=create_generator(), config=create_config(), batch_size=1,
                                      vocabulary_map=create_vocabulary_maps()):
    return xainano_sequence_generator(generator, config, token_parser, batch_size, vocabulary_map[0], augmentor, post_processor)

def image_sequencer(batch_size, image_map, base, vocabulary_map, augmentor, split=(0, 80)):
    vocabulary_size = len(vocabulary_map)
    no_images = len(image_map)
    index = int((split[0] / 100) * no_images)
    while True:
        max_width = int(0)
        max_height = int(0)
        inputs = []
        input_sequences = []
        targets = []
        max_seq_len = int(0)
        for batch_index in range(batch_size):
            if index >= int((split[1] / 100) * no_images):
                index = int((split[0] / 100) * no_images)
            fname, formula = image_map[index]

            image = read_img(os.path.join(base, "images", fname))
            image = augmentor.size_changing_augment(image)
            image = augmentor.augment(image)
            max_width = max(utils.w(image), max_width)
            max_height = max(utils.h(image), max_height)


            tokens = list(formula)
            input_sequence = list(tokens)
            input_sequence.insert(0, "<start>")
            tokens.append("<end>")

            inputs.append(image)
            input_sequences.append(np.array(input_sequence))
            targets.append(np.array(tokens))

            max_seq_len = max(max_seq_len, len(tokens))

            index += 1

        for batch_index in range(batch_size):
            image = inputs[batch_index]
            rw, rh = max_width - utils.w(image), max_height - utils.h(image)
            left, top = int(rw / 2), int(rh / 2)
            inputs[batch_index] = utils.pad_image(image, top, left, rh - top, rw - left)

            # Resize sequence
            seq = input_sequences[batch_index]
            input_sequences[batch_index] = np.append(seq, np.repeat('<end>', max_seq_len - len(seq)))
            input_sequences[batch_index] = [vocabulary_map[token] for token in input_sequences[batch_index]]

            t_seq = targets[batch_index]
            targets[batch_index] = np.append(t_seq, np.repeat('<end>', max_seq_len - len(t_seq)))
            targets[batch_index] = [vocabulary_map[token] for token in targets[batch_index]]

        yield [np.stack(inputs), to_categorical(input_sequences, vocabulary_size)], \
              to_categorical(targets, vocabulary_size)

def tar_image_sequencer(batch_size, tar_file, image_map, vocabulary_map, augmentor, split=(0, 80)):
    tar_folder = tar_file.getmembers()[0]
    vocabulary_size = len(vocabulary_map)
    no_images = len(image_map)
    index = int((split[0] / 100) * no_images)
    while True:
        max_width = int(0)
        max_height = int(0)
        inputs = []
        input_sequences = []
        targets = []
        max_seq_len = int(0)
        for batch_index in range(batch_size):

            # Try loading file, until one is found
            image_file = None
            while image_file is None:
                if index >= int((split[1] / 100) * no_images):
                    index = int((split[0] / 100) * no_images)
                fname, formula = image_map[index]
                try:
                    image_file = tar_file.extractfile(os.path.join(tar_folder, fname))
                except KeyError:
                    index += 1

            image = image_from_bytes(image_file.read())
            image_file.close()

            image = augmentor.augment(image)
            max_width = max(utils.w(image), max_width)
            max_height = max(utils.h(image), max_height)

            tokens = list(formula)
            input_sequence = list(tokens)
            input_sequence.insert(0, "<start>")
            tokens.append("<end>")

            inputs.append(image)
            input_sequences.append(np.array(input_sequence))
            targets.append(np.array(tokens))

            max_seq_len = max(max_seq_len, len(tokens))

            index += 1

        for batch_index in range(batch_size):
            image = inputs[batch_index]
            rw, rh = max_width - utils.w(image), max_height - utils.h(image)
            left, top = int(rw / 2), int(rh / 2)
            inputs[batch_index] = utils.pad_image(image, top, left, rh - top, rw - left)

            # Resize sequence
            seq = input_sequences[batch_index]
            input_sequences[batch_index] = np.append(seq, np.repeat('<end>', max_seq_len - len(seq)))
            input_sequences[batch_index] = [vocabulary_map[token] for token in input_sequences[batch_index]]

            t_seq = targets[batch_index]
            targets[batch_index] = np.append(t_seq, np.repeat('<end>', max_seq_len - len(t_seq)))
            targets[batch_index] = [vocabulary_map[token] for token in targets[batch_index]]

        yield [np.stack(inputs), to_categorical(input_sequences, vocabulary_size)], \
              to_categorical(targets, vocabulary_size)
