import file_utils as utils
from trainer import model
from utilities import parse_arg
from numpy.random import seed
from os import path
from graphics import augment
import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score

from tensorflow import set_random_seed
from trainer.sequence import predefined_image_sequence_generator
from trainer.defaults import *
from trainer.predictor import create_predictor
from trainer.sequence import create_parser

seed(1337)
set_random_seed(1337)

weights_fname = parse_arg('--weights', '/Users/balazs/university/models/model-att1-conv64-rowlstm/weights_19.h5')
data_base_dir = parse_arg('--data-base-dir', '/Users/balazs/university/model')

max_length = 200

vocabulary = utils.read_pkl(path.join(data_base_dir, "vocabulary.pkl"))
vocabulary = vocabulary | {"<start>", "<end>", "^", "_", "\\frac", "{", "}", "\\mbox", "\\to", "\\left"} \
                        | {"\\right", "\\cdots"}
parser = create_parser(vocabulary)
vocabulary = sorted(vocabulary)
vocabulary_maps = create_vocabulary_maps(vocabulary)

model, encoder, decoder = model.create_default(len(vocabulary), None)

if not utils.file_exists(weights_fname):
    print("weights file does not exist: " + weights_fname)
    exit(1)

weights = utils.read_npy(weights_fname)
model.set_weights(weights)

images = utils.read_pkl(path.join(data_base_dir, "data_test_2014.pkl"))

predict = create_predictor(encoder, decoder, vocabulary, vocabulary_maps[0], vocabulary_maps[1], max_length)
augmentor = augment.Augmentor()

def wer(r, h):
    """
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : list
    h : list

    Returns
    -------
    int

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    1
    >>> wer("who is there".split(), "".split())
    3
    >>> wer("".split(), "who is there".split())
    3
    """
    # initialisation
    import numpy
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]

def exp_rate(truth, predicted):
    if len(truth) > len(predicted):
        predicted = np.append(predicted, np.repeat("<end>", len(truth) - len(predicted)))
    elif len(predicted) > len(truth):
        truth = np.append(truth, np.repeat("<end>", len(predicted) - len(truth)))

    predicted = np.array(predicted)
    truth = np.array(truth)
    score = accuracy_score(predicted, truth)

    return score

num = 0
total_wer = 0
total_exp_rate = 0

for image, truth in images:
    grayscale_image = augmentor.grayscale(image)

    predicted, predicted_parsed = predict(grayscale_image)
    if len(predicted_parsed) >= max_length:
        predicted_parsed = predicted_parsed[:len(truth) + 5]

    truth = parser.parse(truth)
    truth = list(filter(lambda a: a != " ", truth))

    total_wer += wer(truth, predicted_parsed)
    total_exp_rate += exp_rate(truth, predicted_parsed)
    num += 1

avg_wer = total_wer / num
avg_exp_rate = total_exp_rate / num

print(avg_wer)
print(avg_exp_rate)