from transformer import t2t_transformer

import tensorflow as tf
import numpy as np


def make_transformer_model_fn(hparams,
                              decode_hparams=None,
                              use_tpu=False):
    """ Makes a transformer model_fn for usage with Estimator. """
    def wrapping_model_fn(features, labels, mode, params=None, config=None):
        return t2t_transformer.Transformer.estimator_model_fn(
            hparams,
            features,
            labels,
            mode,
            config=config,
            params=params,
            decode_hparams=decode_hparams,
            use_tpu=use_tpu)
    return wrapping_model_fn


def make_input_fn():
    input = {
        "inputs": np.array([[1,2,3], [4,5,6], [1,2,4], [6,7,8]]),
        "targets": np.array([[1,2,3], [4,5,6], [1,2,4], [6,7,8]])
    }
    return tf.estimator.inputs.numpy_input_fn(
        input,
        batch_size=128,
        num_epochs=1,
        shuffle=True
    )


def train():
    hparams = t2t_transformer.transformer_base()
    model_fn = make_transformer_model_fn(hparams)
    run_config = tf.estimator.RunConfig()
    run_config.data_parallelism = False
    estimator = tf.estimator.Estimator(model_fn, config=run_config)
    estimator.train(make_input_fn())


train()