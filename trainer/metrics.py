import keras.backend as K
import keras


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
