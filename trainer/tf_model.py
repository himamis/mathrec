from trainer.metrics import *

import tensorflow as tf
import trainer.tf_initializers as tfi
import trainer.default_type as t


def default_cnn_block(**kwargs):
    filter_size = kwargs['filter_size']
    kernel_init = kwargs['kernel_init']
    bias_init = kwargs['bias_init']
    conv = kwargs['conv']
    activation = kwargs['activation']
    is_training = kwargs['is_training']
    summarize = kwargs['summarize']
    w_1 = tf.get_variable("kernel_1", shape=(3, 3, kwargs['prev_filter_size'], filter_size),
                          initializer=kernel_init, dtype=t.my_tf_float)
    b_1 = tf.get_variable("bias_1", shape=[filter_size], initializer=bias_init,
                          dtype=t.my_tf_float)

    w_2 = tf.get_variable("kernel_2", shape=(3, 3, filter_size, filter_size),
                          initializer=kernel_init, dtype=t.my_tf_float)
    b_2 = tf.get_variable("bias_2", shape=[filter_size], initializer=bias_init,
                          dtype=t.my_tf_float)

    conv_1 = tf.nn.conv2d(conv, w_1, strides=[1, 1, 1, 1], padding='SAME') + b_1
    act_1 = activation(conv_1)
    conv_2 = tf.nn.conv2d(act_1, w_2, strides=[1, 1, 1, 1], padding='SAME') + b_2
    bn = tf.layers.batch_normalization(conv_2, training=is_training,
                                       name="batch_norm_{}".format(filter_size))
    act_2 = activation(bn)

    if summarize:
        tf.summary.histogram('weights_1', w_1)
        tf.summary.histogram('weights_2', w_2)

        tf.summary.histogram('biases_1', b_1)
        tf.summary.histogram('biases_2', b_2)

        tf.summary.histogram('activations_1', act_1)
        tf.summary.histogram('activations_2', act_2)

    return act_2


def dense_cnn_block_creator(dense_size=4, dropout=0.2):

    def dense_cnn_block(**kwargs):
        filter_size = kwargs['filter_size']
        kernel_init = kwargs['kernel_init']
        bias_init = kwargs['bias_init']
        conv = kwargs['conv']
        activation = kwargs['activation']
        is_training = kwargs['is_training']
        summarize = kwargs['summarize']
        size = kwargs['prev_filter_size']
        is_last_block = kwargs['last_block']
        for i in range(dense_size):
            with tf.variable_scope("layer_{}".format(i)):
                w = tf.get_variable("kernel", shape=(3, 3, size, filter_size),
                                    initializer=kernel_init, dtype=t.my_tf_float)
                b = tf.get_variable("bias", shape=[filter_size], initializer=bias_init,
                                    dtype=t.my_tf_float)
                size = filter_size
                conv = tf.nn.conv2d(conv, w, strides=[1, 1, 1, 1], padding='SAME') + b
                bn = tf.layers.batch_normalization(conv, training=is_training,
                                                   name="batch_norm_{}".format(filter_size))
                conv = activation(bn)
                if is_last_block and dropout is not None:
                    conv = tf.layers.dropout(conv, rate=dropout, training=is_training)

                if summarize:
                    tf.summary.histogram('kernel', w)
                    tf.summary.histogram('bias', b)
                    tf.summary.histogram('activation', conv)

        return conv

    return dense_cnn_block


class AttentionWrapper(tf.nn.rnn_cell.RNNCell):

    def __init__(self, cell, att_dim, units, feature_grids, image_masks, dense_initializer, dense_bias_initializer):
        super(AttentionWrapper, self).__init__()
        self.cell = cell
        self.feature_grid_dims = [feature_grid.shape[3] for feature_grid in feature_grids]
        self.feature_grids = feature_grids
        self.image_masks = image_masks
        self.attention_us = []
        self.attention_u_bs = []
        self.attention_v_as = []
        self.attention_v_a_bs = []
        self.watch_vectors = []
        for index, (feature_grid, feature_grid_dim) in enumerate(zip(feature_grids, self.feature_grid_dims)):
            attention_u = tf.get_variable(name="decoder_attention_u_scale_{}".format(index),
                                          initializer=dense_initializer,
                                          shape=[feature_grid_dim, att_dim], dtype=t.my_tf_float)
            attention_u_b = tf.get_variable(name="decoder_attention_u_b_scale_{}".format(index),
                                            initializer=dense_bias_initializer,
                                            shape=[att_dim], dtype=t.my_tf_float)

            attention_v_a = tf.get_variable(name="decoder_attention_v_a_scale_{}".format(index),
                                            initializer=dense_initializer,
                                            shape=[att_dim, 1], dtype=t.my_tf_float)
            attention_v_a_b = tf.get_variable(name="decoder_attention_v_a_b_scale_{}".format(index),
                                              initializer=dense_bias_initializer,
                                              shape=[1], dtype=t.my_tf_float)

            # Can be precomputed
            watch_vector = tf.tensordot(feature_grid, attention_u, axes=1) + attention_u_b  # [batch, h, w, dim_attend]

            self.attention_us.append(attention_u)
            self.attention_u_bs.append(attention_u_b)
            self.attention_v_as.append(attention_v_a)
            self.attention_v_a_bs.append(attention_v_a_b)
            self.watch_vectors.append(watch_vector)

        self.attention_w = tf.get_variable(name="decoder_attention_w",
                                      initializer=dense_initializer,
                                      shape=[units, att_dim], dtype=t.my_tf_float)
        self.attention_w_b = tf.get_variable(name="decoder_attention_w_b",
                                        initializer=dense_bias_initializer,
                                        shape=[att_dim], dtype=t.my_tf_float)

    @property
    def wrapped_cell(self):
        return self.cell

    @property
    def state_size(self):
        return self.cell.state_size

    @property
    def output_size(self):
        return self.cell.output_size

    def zero_state(self, batch_size, dtype):
        return self.cell.zero_state(batch_size, dtype)

    def __call__(self, inputs, state, scope=None):
        h_tm1, c_tm1 = state
        with tf.name_scope("ctx"):
            # context vector
            speller_vector = tf.tensordot(h_tm1, self.attention_w, axes=1) + self.attention_w_b
            ctxs = []
            for watch_vector, attention_v_a, attention_v_a_b, feature_grid in \
                    zip(self.watch_vectors, self.attention_v_as, self.attention_v_a_bs, self.feature_grids):
                tanh_vector = tf.tanh(watch_vector + speller_vector[:, None, None, :])  # [batch, h, w, dim_attend]
                e_ti = tf.tensordot(tanh_vector, attention_v_a, axes=1) + attention_v_a_b  # [batch, h, w, 1]
                alpha = tf.exp(e_ti)
                alpha = alpha * self.image_masks
                alpha = alpha / tf.reduce_sum(alpha, axis=[1, 2], keepdims=True)
                # masked_e_ti = e_ti * image_masks
                # alpha = tf.nn.softmax(masked_e_ti)
                ctx = tf.reduce_sum(feature_grid * alpha, axis=[1, 2])
                ctxs.append(ctx)

            if len(ctxs) == 1:
                ctx = ctxs[0]
            else:
                ctx = tf.concat(ctxs, axis=1)
        return self.cell(tf.concat([inputs, ctx], 1), state, scope=scope)

class CNNEncoder:

    def __init__(self,
                 filter_sizes=None,
                 kernel_init=None,
                 bias_init=None,
                 activation=None,
                 cnn_block=None):
        if filter_sizes is None:
            filter_sizes = [64, 128, 256, 512]
        self.filter_sizes = filter_sizes
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.activation = activation
        self.cnn_block = cnn_block
        if cnn_block is None:
            raise ValueError("cnn_block must not be None")

    def __call__(self, input_images, image_mask, is_training, summarize=False):
        convolutions = []
        conv = (input_images - 128) / 128
        prev_size = 1
        for index, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("conv_block_{}_{}".format(filter_size, index)):
                conv = self.cnn_block(conv=conv, filter_size=filter_size, prev_filter_size=prev_size,
                                      bias_init=self.bias_init, kernel_init=self.kernel_init,
                                      activation=self.activation, is_training=is_training, summarize=summarize,
                                      last_block=index+1 == len(self.filter_sizes))
                prev_size = filter_size
                conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                      name='max_pool_{}'.format(filter_size))
                image_mask = image_mask[:, 0::2, 0::2]

            convolutions.append(conv)

        return convolutions, image_mask


class RowEncoder:

    def __init__(self, encoder_size=512, kernel_init=None, recurrent_init=None, bidirectional=True):
        self.encoder_size = encoder_size
        self.kernel_init = kernel_init
        self.recurrent_init = recurrent_init
        self.bidirectional = bidirectional

    def __call__(self, feature_grid, image_mask, summarize=False):
        with tf.name_scope("row_encoder"):
            cell_fw = tf.nn.rnn_cell.LSTMCell(self.encoder_size, initializer=self.kernel_init, dtype=t.my_tf_float)

            if self.bidirectional:
                cell_bw = tf.nn.rnn_cell.LSTMCell(self.encoder_size, initializer=self.kernel_init, dtype=t.my_tf_float)

            masked_feature_grid = feature_grid * image_mask

            def apply_fun(image_row):
                if self.bidirectional:
                    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, image_row, dtype=t.my_tf_float)
                    outputs = tf.concat(outputs, 2)
                else:
                    outputs, _ = tf.nn.dynamic_rnn(cell_fw, image_row, dtype=t.my_tf_float)
                return outputs

            height_first = tf.transpose(masked_feature_grid, [1, 0, 2, 3])
            output = tf.map_fn(apply_fun, height_first)
            batch_first = tf.transpose(output, [1, 0, 2, 3])

            if summarize:
                tf.summary.histogram("feature_grid_encoder", feature_grid)
                tf.summary.histogram("masked_grid_encoder", masked_feature_grid)
                tf.summary.histogram("height_f", height_first)
                tf.summary.histogram("batch_first", batch_first)

        return batch_first


class AttentionDecoder:

    def __init__(self, vocabulary_size,
                 embedding_dim=256,
                 units=512, att_dim=512,
                 lstm_kernel_initializer=None,
                 lstm_bias_initializer=None,
                 dense_initializer=None,
                 dense_bias_initializer=None,
                 lstm_recurrent_kernel_initializer=None):
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.units = units
        self.att_dim = att_dim
        self.lstm_kernel_initializer = lstm_kernel_initializer
        self.lstm_bias_initializer = lstm_bias_initializer
        self.dense_initializer = dense_initializer
        self.dense_bias_initializer = dense_bias_initializer
        self.lstm_recurrent_kernel_initializer = lstm_recurrent_kernel_initializer

    def __call__(self, feature_grid, image_masks, inputs, init_h, init_c, summarize=False):
        feature_grids = tf.unstack(feature_grid)

        cell = AttentionWrapper(tf.nn.rnn_cell.LSTMCell(self.units), self.att_dim, self.units, feature_grids,
                                image_masks, self.dense_initializer, self.dense_bias_initializer)
        outputs, states = tf.nn.dynamic_rnn(cell, inputs, dtype=t.my_tf_float,
                                            initial_state=tf.nn.rnn_cell.LSTMStateTuple(init_c, init_h))

        return [outputs, states[1], states[0]]


class Model:

    def __init__(self, vocabulary_size, encoder_size=512,
                 filter_sizes=None,
                 decoder_units=512,
                 attention_dim=512,
                 embedding_dim=256,
                 conv_kernel_init=tfi.he_normal(dtype=t.my_tf_float),
                 conv_bias_init=tf.initializers.zeros(dtype=t.my_tf_float),
                 conv_activation=tf.nn.relu,
                 cnn_block=default_cnn_block,
                 encoder_kernel_init=tf.initializers.orthogonal(dtype=t.my_tf_float),
                 decoder_kernel_init=tf.initializers.orthogonal(dtype=t.my_tf_float),
                 decoder_recurrent_kernel_init=tf.initializers.orthogonal(dtype=t.my_tf_float),
                 decoder_bias_init=tf.initializers.zeros(dtype=t.my_tf_float),
                 dense_init=tfi.glorot_normal(dtype=t.my_tf_float),
                 dense_bias_init=tf.initializers.zeros(dtype=t.my_tf_float),
                 bidirectional=True,
                 multi_scale_attention=False):
        self.vocabulary_size = vocabulary_size
        self.dense_init = dense_init
        self.dense_bias_init = dense_bias_init
        self.multi_scale_attention = multi_scale_attention
        if filter_sizes is None:
            filter_sizes = [64, 128, 256, 512]
        self.decoder_units = decoder_units
        self.embedding_dim = embedding_dim
        self._encoder = CNNEncoder(
            filter_sizes=filter_sizes,
            kernel_init=conv_kernel_init,
            bias_init=conv_bias_init,
            activation=conv_activation,
            cnn_block=cnn_block
        )
        self._row_encoder = RowEncoder(
            encoder_size=encoder_size,
            kernel_init=encoder_kernel_init,
            recurrent_init=decoder_recurrent_kernel_init,
            bidirectional=bidirectional
        )
        if multi_scale_attention:
            self._row_encoder_scale = RowEncoder(
                encoder_size=int(encoder_size / 2),
                kernel_init=encoder_kernel_init,
                bidirectional=bidirectional
            )
        self._decoder = AttentionDecoder(
            vocabulary_size=vocabulary_size,
            units=decoder_units,
            att_dim=attention_dim,
            embedding_dim=embedding_dim,
            lstm_kernel_initializer=decoder_kernel_init,
            lstm_bias_initializer=decoder_bias_init,
            dense_initializer=dense_init,
            dense_bias_initializer=dense_bias_init,
            lstm_recurrent_kernel_initializer=decoder_recurrent_kernel_init
        )

    def feature_grid(self, input_images, input_image_masks, is_training, summarize=False):
        with tf.variable_scope("convolutional_encoder", reuse=tf.AUTO_REUSE):
            encoded_images, image_masks = self._encoder(input_images=input_images, image_mask=input_image_masks,
                                           is_training=is_training, summarize=summarize)

        feature_grid = []
        with tf.variable_scope("row_encoder", reuse=tf.AUTO_REUSE):
            re_encoded_images = self._row_encoder(feature_grid=encoded_images[-1], image_mask=image_masks,
                                                  summarize=summarize)

        feature_grid.append(re_encoded_images)

        if self.multi_scale_attention:
            with tf.variable_scope("multi_scale_row_encoder", reuse=tf.AUTO_REUSE):
                re_encoded_images_scale = self._row_encoder_scale(feature_grid=encoded_images[-2])
            feature_grid.append(re_encoded_images_scale)

        return tf.stack(feature_grid), image_masks

    def calculate_decoder_init(self, feature_grid, image_masks):
        feature_grids = tf.unstack(feature_grid)
        with tf.variable_scope("decoder_initializer", reuse=tf.AUTO_REUSE):
            encoded_mean = tf.reduce_sum(feature_grids[-1] * image_masks, axis=[1, 2]) / \
                           tf.reduce_sum(image_masks, axis=[1, 2])
            calculate_h0 = tf.layers.dense(encoded_mean, use_bias=True, activation=tf.nn.tanh,
                                           units=self.decoder_units)
            calculate_c0 = tf.layers.dense(encoded_mean, use_bias=True, activation=tf.nn.tanh,
                                           units=self.decoder_units)

        return calculate_h0, calculate_c0

    def decoder_init(self, batch_size):
        init_h = tf.placeholder(dtype=t.my_tf_float, shape=(batch_size, self.decoder_units))
        init_c = tf.placeholder(dtype=t.my_tf_float, shape=(batch_size, self.decoder_units))

        return init_h, init_c

    def decoder(self, feature_grid, image_masks, input_characters, init_h, init_c, summarize=False):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            embedding = tf.get_variable(name="embedding", initializer=tf.initializers.random_normal, dtype=t.my_tf_float,
                                        shape=[self.vocabulary_size, self.embedding_dim])
            embedded_characters = tf.nn.embedding_lookup(embedding, input_characters)
            outputs, state_h, state_c = self._decoder(feature_grid=feature_grid, image_masks=image_masks,
                                                      inputs=embedded_characters, init_h=init_h, init_c=init_c,
                                                      summarize=summarize)

            w = tf.get_variable(name="output_w", initializer=self.dense_init,
                                shape=[self.decoder_units, self.vocabulary_size],
                                dtype=t.my_tf_float)
            b = tf.get_variable(name="output_b", initializer=self.dense_bias_init,
                                shape=[self.vocabulary_size], dtype=t.my_tf_float)

            output = tf.tensordot(outputs, w, axes=1) + b

        return [state_h, state_c] + [output]

    def training(self, input_images, input_image_masks, input_characters):
        feature_grid, image_masks = self.feature_grid(input_images, input_image_masks, is_training=True, summarize=True)

        tf.summary.histogram("feature_grid", feature_grid)

        calculate_h, calculate_c = self.calculate_decoder_init(feature_grid, image_masks)

        outputs = self.decoder(feature_grid, image_masks, input_characters, calculate_h, calculate_c,
                               summarize=True)

        tf.summary.histogram("activation_state_h", outputs[0])

        return outputs[-1]

    def sample(self, feature_grid, previous_character, init_h, init_c):
        outputs = self.decoder(feature_grid, previous_character, init_h, init_c)
        return outputs
