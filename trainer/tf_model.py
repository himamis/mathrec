from trainer.metrics import *

import tensorflow as tf
import trainer.tf_initializers as tfi


class CNNEncoder:

    def __init__(self,
                 filter_sizes=None,
                 kernel_init=None,
                 bias_init=None,
                 activation=None):
        if filter_sizes is None:
            filter_sizes = [64, 128, 256, 512]
        self.filter_sizes = filter_sizes
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.activation = activation

    def __call__(self, input_images, is_training=True):
        convolutions = []
        conv = (input_images - 128) / 128
        for index, filter_size in enumerate(self.filter_sizes):
            conv = tf.layers.conv2d(conv, filters=filter_size, kernel_size=3, strides=1, padding='same',
                                    kernel_initializer=self.kernel_init, bias_initializer=self.bias_init,
                                    activation=self.activation,
                                    name="conv_block_{}_filter_{}_1".format(index, filter_size))
            conv = tf.layers.conv2d(conv, filters=filter_size, kernel_size=3, strides=1, padding='same',
                                    kernel_initializer=self.kernel_init, bias_initializer=self.bias_init,
                                    activation=None,
                                    name="conv_block_{}_filter_{}_2".format(index, filter_size))
            conv = tf.layers.batch_normalization(conv, training=is_training,
                                                 name="batch_norm_block_{}_filter_{}".format(index, filter_size))
            conv = tf.nn.relu(conv, name="relu_block_{}_filter_{}".format(index, filter_size))
            conv = tf.layers.max_pooling2d(conv, pool_size=2, strides=2, padding='valid',
                                           name="max_pooling_{}_filter_{}".format(index, filter_size))
            convolutions.append(conv)

        return None, convolutions


class RowEncoder:

    def __init__(self, encoder_size=512, kernel_init=None, bidirectional=True):
        self.encoder_size = encoder_size
        self.kernel_init = kernel_init
        self.bidirectional = bidirectional

    def __call__(self, feature_grid):
        encoder = tf.keras.layers.LSTM(self.encoder_size, kernel_initializer=self.kernel_init,
                                       return_sequences=True, name="row_encoder_lstm")
        if self.bidirectional:
            encoder = tf.keras.layers.Bidirectional(encoder)

        def step(x, _):
            output = encoder(x)
            return output, []

        _, outputs, _ = K.rnn(step, feature_grid, [])

        return None, outputs


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

    def __call__(self, feature_grids, inputs):
        feature_grid_dims = [feature_grid.shape[3] for feature_grid in feature_grids]
        batch_size = feature_grids[0].shape[0]
        h_init = tf.placeholder(shape=[batch_size, self.units], dtype=tf.float32,
                                name='decoder_lstm_kernel_h_init_state')
        c_init = tf.placeholder(shape=[batch_size, self.units], dtype=tf.float32,
                                name='decoder_lstm_kernel_c_init_state')
        kernel = tf.get_variable(name="decoder_lstm_kernel_{}".format(self.units),
                                 initializer=self.lstm_kernel_initializer,
                                 shape=[self.embedding_dim, 4 * self.units])
        recurrent_kernel = tf.get_variable(name="decoder_lstm_recurrent_kernel_{}".format(self.units),
                                           initializer=self.lstm_recurrent_kernel_initializer,
                                           shape=[self.units, 4 * self.units])
        context_kernel = tf.get_variable(name="decoder_lstm_context_kernel_{}".format(self.units),
                                         initializer=self.lstm_recurrent_kernel_initializer,
                                         shape=[sum(feature_grid_dims), 4 * self.units])
        bias = tf.get_variable(name="decoder_lstm_bias_{}".format(self.units),
                               initializer=self.lstm_bias_initializer,
                               shape=[4 * self.units])
        attention_us = []
        attention_u_bs = []
        attention_v_as = []
        attention_v_a_bs = []
        watch_vectors = []
        for index, (feature_grid, feature_grid_dim) in enumerate(zip(feature_grids, feature_grid_dims)):
            attention_u = tf.get_variable(name="decoder_attention_u_scale_{}".format(index),
                                          initializer=self.dense_initializer,
                                          shape=[feature_grid_dim, self.att_dim])
            attention_u_b = tf.get_variable(name="decoder_attention_u_b_scale_{}".format(index),
                                            initializer=self.dense_bias_initializer,
                                            shape=[self.att_dim])

            attention_v_a = tf.get_variable(name="decoder_attention_v_a_scale_{}".format(index),
                                            initializer=self.dense_initializer,
                                            shape=[self.att_dim, 1])
            attention_v_a_b = tf.get_variable(name="decoder_attention_v_a_b_scale_{}".format(index),
                                              initializer=self.dense_bias_initializer,
                                              shape=[1])

            # Can be precomputed
            watch_vector = tf.tensordot(feature_grid, attention_u, axes=1) + attention_u_b  # [batch, h, w, dim_attend]

            attention_us.append(attention_u)
            attention_u_bs.append(attention_u_b)
            attention_v_as.append(attention_v_a)
            attention_v_a_bs.append(attention_v_a_b)
            watch_vectors.append(watch_vector)

        attention_w = tf.get_variable(name="decoder_attention_w",
                                      initializer=self.dense_initializer,
                                      shape=[self.units, self.att_dim])
        attention_w_b = tf.get_variable(name="decoder_attention_w_b",
                                        initializer=self.dense_bias_initializer,
                                        shape=[self.att_dim])

        def step(state, input):
            h_tm1, c_tm1 = tf.unstack(state)

            x_i = tf.nn.bias_add(tf.matmul(input, kernel[:, :self.units]), bias[:self.units])
            x_f = tf.nn.bias_add(tf.matmul(input, kernel[:, self.units:self.units * 2]), bias[self.units:self.units * 2])
            x_c = tf.nn.bias_add(tf.matmul(input, kernel[:, self.units * 2:self.units * 3]), bias[self.units * 2:self.units * 3])
            x_o = tf.nn.bias_add(tf.matmul(input, kernel[:, self.units * 3:]), bias[self.units * 3:])

            r_i = tf.tensordot(h_tm1, recurrent_kernel[:, :self.units], 1)
            r_f = tf.tensordot(h_tm1, recurrent_kernel[:, self.units:self.units * 2], 1)
            r_c = tf.tensordot(h_tm1, recurrent_kernel[:, self.units * 2:self.units * 3], 1)
            r_o = tf.tensordot(h_tm1, recurrent_kernel[:, self.units * 3:], 1)

            # context vector
            speller_vector = tf.tensordot(h_tm1, attention_w, axes=1) + attention_w_b
            ctxs = []
            for watch_vector, attention_v_a, attention_v_a_b, feature_grid in \
                    zip(watch_vectors, attention_v_as, attention_v_a_bs, feature_grids):
                tanh_vector = tf.tanh(watch_vector + speller_vector[:, None, None, :])  # [batch, h, w, dim_attend]
                e_ti = tf.tensordot(tanh_vector, attention_v_a, axes=1) + attention_v_a_b  # [batch, h, w, 1]
                alpha = tf.exp(e_ti)
                alpha = tf.squeeze(alpha, axis=3)
                alpha = alpha / tf.reduce_sum(alpha, axis=[1, 2], keepdims=True)
                ctx = tf.reduce_sum(feature_grid * alpha[:, :, :, None], axis=[1, 2])
                ctxs.append(ctx)

            ctx = tf.concat(ctxs, axis=1)
            c_i = tf.matmul(ctx, context_kernel[:, :self.units])
            c_f = tf.matmul(ctx, context_kernel[:, self.units:self.units * 2])
            c_c = tf.matmul(ctx, context_kernel[:, self.units * 2:self.units * 3])
            c_o = tf.matmul(ctx, context_kernel[:, self.units * 3:])

            i = tf.keras.backend.hard_sigmoid(x_i + r_i + c_i)
            f = tf.keras.backend.hard_sigmoid(x_f + r_f + c_f)
            c = f * c_tm1 + i * tf.tanh(x_c + r_c + c_c)
            o = tf.keras.backend.hard_sigmoid(x_o + r_o + c_o)

            h = o * tf.tanh(c)

            return tf.stack([h, c])

        states = tf.scan(step,
                         tf.transpose(inputs, [1, 0, 2]),
                         initializer=tf.stack([h_init, c_init]))

        hs, cs = tf.unstack(tf.transpose(states, [1, 2, 0, 3]))

        return [h_init, c_init], [hs, cs]


class Model:

    def __init__(self, vocabulary_size, encoder_size=512,
                 filter_sizes=None,
                 decoder_units=512,
                 attention_dim=512,
                 embedding_dim=256,
                 conv_kernel_init=tfi.he_normal(),
                 conv_bias_init=tf.initializers.zeros(),
                 conv_activation=tf.nn.relu,
                 encoder_kernel_init=tf.initializers.orthogonal(),
                 decoder_kernel_init=tf.initializers.orthogonal(),
                 decoder_bias_init=tf.initializers.zeros(),
                 dense_init=tfi.glorot_normal(),
                 dense_bias_init=tf.initializers.zeros(),
                 bidirectional=True,
                 multi_scale_attention=False):
        self.vocabulary_size = vocabulary_size
        self.dense_init = dense_init
        self.dense_bias_init = dense_bias_init
        self.multi_scale_attention = multi_scale_attention
        if filter_sizes is None:
            filter_sizes = [64, 128, 256, 512]
        self.embedding_dim = embedding_dim
        self.encoder = CNNEncoder(
            filter_sizes=filter_sizes,
            kernel_init=conv_kernel_init,
            bias_init=conv_bias_init,
            activation=conv_activation
        )
        self.row_encoder = RowEncoder(
            encoder_size=encoder_size,
            kernel_init=encoder_kernel_init,
            bidirectional=bidirectional
        )
        if multi_scale_attention:
            self.row_encoder_scale = RowEncoder(
                encoder_size=int(encoder_size / 2),
                kernel_init=encoder_kernel_init,
                bidirectional=bidirectional
            )
        self.decoder = AttentionDecoder(
            vocabulary_size=vocabulary_size,
            units=decoder_units,
            att_dim=attention_dim,
            lstm_kernel_initializer=decoder_kernel_init,
            lstm_bias_initializer=decoder_bias_init,
            dense_initializer=dense_init,
            dense_bias_initializer=dense_bias_init,
            lstm_recurrent_kernel_initializer=encoder_kernel_init
        )

    def __call__(self, is_training=True, batch_size=32):
        input_images = tf.placeholder(tf.float32, shape=(batch_size, None, None, 1), name="input_images")
        input_characters = tf.placeholder(tf.int32, shape=(batch_size, None))

        embedding = tf.get_variable(name="embedding",
                                    initializer=tf.initializers.random_normal, dtype=tf.float32,
                                    shape=[self.vocabulary_size, self.embedding_dim])
        embedded_characters = tf.nn.embedding_lookup(embedding, input_characters)

        feature_grid = []
        with tf.variable_scope("convolutional_encoder"):
            _, encoded_images = self.encoder(input_images=input_images, is_training=is_training)

        with tf.variable_scope("row_encoder"):
            _, re_encoded_images = self.row_encoder(feature_grid=encoded_images[-1])
            feature_grid.append(re_encoded_images)

        if self.multi_scale_attention:
            with tf.variable_scope("multi_scale_row_encoder"):
                _, re_encoded_images_scale = self.row_encoder_scale(feature_grid=encoded_images[-2])
                feature_grid.append(re_encoded_images_scale)

        with tf.variable_scope("attention_decoder"):
            init, states = self.decoder(feature_grids=feature_grid, inputs=embedded_characters)

        state_h, _ = states
        output = tf.layers.dense(state_h, units=self.vocabulary_size, activation=tf.nn.softmax, use_bias=True,
                                 kernel_initializer=self.dense_init, bias_initializer=self.dense_bias_init,
                                 name="output_dense_softmax")

        # [init_h, init_c], [state_h, state_c, output]
        return [input_images, input_characters] + init, states + [output], embedding
