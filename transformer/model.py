import tensorflow as tf
from transformer import transformer
from transformer import model_params


class TransformerLatex(object):

    def __init__(self, vocabulary_size):
        params = model_params.BASE_PARAMS
        params.update(vocab_size=vocabulary_size)

        with tf.variable_scope("transformer", reuse=tf.AUTO_REUSE):
            self.transformer = transformer.Transformer(params, True)
            self.transformer_evaluate = transformer.Transformer(params, False)

    def __call__(self, inputs, bounding_box_placeholder, targets=None, train=False):
        with tf.variable_scope("transformer", reuse=tf.AUTO_REUSE):
            if train:
                assert targets is not None, "Targets must not be none when training."
                return self.transformer(inputs, bounding_box_placeholder, targets)
            else:
                return self.transformer_evaluate(inputs, bounding_box_placeholder)