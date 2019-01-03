import keras.backend as K
import keras
import numpy as np
from sklearn.metrics import accuracy_score


def get_masked(mask_value, metric):
    mask_value = K.variable(mask_value)

    def masked(y_true, y_pred):
        # find out which timesteps in `y_true` are not the padding character '#'
        mask = K.all(K.equal(y_true, mask_value), axis=-1)
        mask = 1 - K.cast(mask, K.floatx())

        # multiply metric with the mask
        loss = metric(y_true, y_pred) * mask

        # take average w.r.t. the number of unmasked entries
        return K.sum(loss) / K.sum(mask)
    return masked


def get_masked_categorical_crossentropy(mask):
    return get_masked(mask, K.categorical_crossentropy)


def get_masked_categorical_accuracy(mask):
    return get_masked(mask, keras.metrics.categorical_accuracy)


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
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]


def exp_rate(truth, predicted):
    if len(truth) > len(predicted):
        predicted = np.append(predicted, np.repeat(-1, len(truth) - len(predicted)))
    elif len(predicted) > len(truth):
        truth = np.append(truth, np.repeat(-1, len(predicted) - len(truth)))

    predicted = np.array(predicted)
    truth = np.array(truth)
    score = accuracy_score(predicted, truth)

    return score
