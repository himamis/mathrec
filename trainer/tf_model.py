import tensorflow as tf
import trainer.tf_initializers as tfi
import trainer.default_type as t
from trainer.dense_net_creator import DenseNetCreator, selu, bn_relu
from trainer.tf_summary import gif_summary_v2
from trainer import params


def default_cnn_block(**kwargs):
    filter_size = kwargs['filter_size']
    kernel_init = kwargs['kernel_init']
    bias_init = kwargs['bias_init']
    conv = kwargs['conv']
    activation = kwargs['activation']
    is_training = kwargs['is_training']
    summarize = kwargs['summarize']
    w_1 = tf.get_variable("kernel_1", shape=(3, 3, kwargs['prev_filter_size'], filter_size), dtype=t.my_tf_float)
    b_1 = tf.get_variable("bias_1", shape=[filter_size], dtype=t.my_tf_float)

    w_2 = tf.get_variable("kernel_2", shape=(3, 3, filter_size, filter_size), dtype=t.my_tf_float)
    b_2 = tf.get_variable("bias_2", shape=[filter_size], dtype=t.my_tf_float)

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


# VGG-net style
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

    def __init__(self, cell, att_dim, units, feature_grid, image_masks, dense_initializer, dense_bias_initializer,
                 input_images=None, summarize=False):
        super(AttentionWrapper, self).__init__()
        self.cell = cell
        self.feature_grid_dim = feature_grid.shape[1] if params.data_format != 'channels_last' else feature_grid.shape[3]
        self.feature_grid = feature_grid
        self.image_masks = image_masks
        self.att_dim = att_dim
        self.conv_einsum = 'bhwc,cf->bhwf'
        self.conv_features = [1, 2]
        if params.data_format != 'channels_last':
            self.conv_einsum = 'bchw,cf->bfhw'
            self.conv_features = [2, 3]

        self.u_f = tf.get_variable(name="u_f", initializer=tf.initializers.random_normal,
                                   shape=(att_dim, att_dim), dtype=t.my_tf_float)
        self.u_f_b = tf.get_variable(name="U_f_b", initializer=tf.initializers.zeros,
                                     shape=(att_dim,), dtype=t.my_tf_float)

        self.attention_u = tf.get_variable(name="decoder_attention_u_scale",
                                           initializer=dense_initializer,
                                           shape=[self.feature_grid_dim, att_dim], dtype=t.my_tf_float)
        self.attention_u_b = tf.get_variable(name="decoder_attention_u_b_scale",
                                             initializer=dense_bias_initializer,
                                             shape=[att_dim], dtype=t.my_tf_float)

        self.attention_v_a = tf.get_variable(name="decoder_attention_v_a_scale",
                                             initializer=dense_initializer,
                                             shape=[att_dim, 1], dtype=t.my_tf_float)
        self.attention_v_a_b = tf.get_variable(name="decoder_attention_v_a_b_scale",
                                               initializer=dense_bias_initializer,
                                               shape=[1], dtype=t.my_tf_float)

        # Can be precomputed
        # [batch, h, w, dim_attend]
        self.watch_vector = tf.einsum(self.conv_einsum, self.feature_grid, self.attention_u) + self.attention_u_b

        self.attention_w = tf.get_variable(name="decoder_attention_w",
                                           initializer=dense_initializer,
                                           shape=[units, att_dim], dtype=t.my_tf_float)
        self.attention_w_b = tf.get_variable(name="decoder_attention_w_b",
                                             initializer=dense_bias_initializer,
                                             shape=[att_dim], dtype=t.my_tf_float)

        self.input_images = input_images
        self.summarize = summarize
        if self.summarize:
            assert self.input_images is not None

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
        if isinstance(state, tf.nn.rnn_cell.LSTMStateTuple):
            h_tm1 = state.h
        elif isinstance(state, list):
            h_tm1 = state[0]
        else:
            h_tm1 = state
        betas = state[1]

        # if self.summarize:
        #     alphas = state[2]

        # Coverage vector
        ft = tf.layers.conv2d(betas, self.att_dim, strides=(1, 1), padding='SAME', data_format=params.data_format,
                              kernel_size=(3, 3))
        coverage_vector = tf.einsum(self.conv_einsum, ft, self.u_f) + self.u_f_b

        # context vector
        speller_vector = tf.matmul(h_tm1, self.attention_w) + self.attention_w_b

        # tanh_vector = tf.tanh(self.watch_vector + speller_vector[:, None, None, :])  # [batch, h, w, dim_attend]
        # [batch] + params.data_format
        tanh_vector = tf.tanh(self.watch_vector + speller_vector[:, None, None, :] + coverage_vector)

        # [batch, h, w, 1]
        e_ti = tf.einsum(self.conv_einsum, tanh_vector, self.attention_v_a) + self.attention_v_a_b
        alpha = tf.exp(e_ti)
        alpha = alpha * self.image_masks
        alpha = alpha / tf.reduce_sum(alpha, axis=self.conv_features, keepdims=True)
        # ctx = tf.reduce_sum(self.feature_grid * betas * alpha, axis=[1, 2])
        ctx = tf.reduce_sum(self.feature_grid * alpha, axis=self.conv_features)
        betas = betas + alpha

        cell_input = tf.concat([inputs, ctx], 1)
        output, new_state = self.cell(cell_input, h_tm1, scope=scope)

        ret_state = [new_state, betas]

        # if self.summarize:
            # resized_alpha = tf.image.resize_area(alpha, tf.shape(self.input_images)[1:3])
            # attention_images = resized_alpha * self.input_images
            # tf.contrib.summary.image("attention_images", attention_images)
            # alpha_image = resized_alpha * 255
            # tf.summary.image("attention", resized_alpha * 255)
            # tf.summary.histogram("alpha", alpha)
            # ret_state.append(alphas)

        return [output, ret_state]


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

    def __call__(self, input_images, image_mask, is_training, summarize=False, **kwargs):
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

        return conv, image_mask


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

    def __call__(self, feature_grid, image_masks, inputs, init_h, init_alphas, summarize=False, input_images=None):
        if params.use_gpu == "n":
            if params.use_new_rnn:
                rnn_cell = tf.contrib.rnn.GRUBlockCellV2(self.units)
            else:
                rnn_cell = tf.nn.rnn_cell.GRUCell(self.units)
        else:
            rnn_cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(self.units)

        rnn_cell = AttentionWrapper(rnn_cell, self.att_dim, self.units, feature_grid,
                                    image_masks, self.dense_initializer, self.dense_bias_initializer,
                                    summarize=summarize, input_images=input_images)

        initial_states = [init_h, init_alphas]
        # if summarize:
        #     zeros = tf.zeros_like(input_images)[:, None, :]
        #     zeros = tf.tile(zeros, [1, tf.shape(inputs)[1], 1, 1, 1])
        #     initial_states.append(zeros)

        outputs, states = tf.nn.dynamic_rnn(rnn_cell, inputs, dtype=t.my_tf_float, initial_state=initial_states)

        # if summarize:
        #     alphas = states[-1]
        #     gif_summary_v2("alphas", alphas, 10, 1)
        #     states = states[:-1]

        return [outputs, states]


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
        self.conv_features = [1, 2]
        if params.data_format != 'channels_last':
            self.conv_features = [2, 3]
        self._encoder = DenseNetCreator(data_format=params.data_format,
                                        efficient=True, growth_rate=12,
                                        include_top=False,
                                        bottleneck=True,
                                        dropout_rate=0.2,
                                        depth=40,
                                        subsample_initial_block=True,
                                        nb_dense_block=3,
                                        activation=bn_relu)
        # self._encoder = CNNEncoder(
        #    filter_sizes=filter_sizes,
        #    kernel_init=conv_kernel_init,
        #    bias_init=conv_bias_init,
        #    activation=conv_activation,
        #    cnn_block=cnn_block
        # )
        #self._row_encoder = RowEncoder(
        #    encoder_size=encoder_size,
        ##    kernel_init=encoder_kernel_init,
        #    recurrent_init=decoder_recurrent_kernel_init,
        #    bidirectional=bidirectional
        #)
        #if multi_scale_attention:
        #    self._row_encoder_scale = RowEncoder(
        #        encoder_size=int(encoder_size / 2),
        #        kernel_init=encoder_kernel_init,
        #        bidirectional=bidirectional
        #    )
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

    def feature_grid(self, input_images, input_image_masks, is_training, r_max, d_max, summarize=False):
        with tf.name_scope("encoder"):
            encoded_images, image_masks = self._encoder(input_images=input_images, image_mask=input_image_masks,
                                                        is_training=is_training, summarize=summarize, r_max=r_max,
                                                        d_max=d_max)

            if summarize:
                tf.summary.histogram('feature_grid', encoded_images)

        return encoded_images, image_masks

    def calculate_decoder_init(self, feature_grid, image_masks):
        with tf.name_scope("decoder_initializer"):
            if image_masks is not None:
                encoded_mean = tf.reduce_sum(feature_grid * image_masks, axis=self.conv_features) / \
                               tf.reduce_sum(image_masks, axis=self.conv_features)
            else:
                encoded_mean = tf.reduce_mean(feature_grid, axis=self.conv_features)
            calculate_h0 = tf.layers.dense(encoded_mean, activation=tf.nn.tanh, units=self.decoder_units)

            one = tf.ones(1, dtype=tf.int32)
            if params.data_format != 'channels_last':
                batch_size = tf.shape(feature_grid[:, -1, -1, -1])
                height_width = tf.shape(feature_grid[-1, -1, :, :])
                alpha_shape = tf.concat([batch_size, one, height_width], axis=0)
            else:
                shape = tf.shape(feature_grid[:, :, :, -1])
                alpha_shape = tf.concat([shape, one], axis=0)
            calculate_alphas = tf.zeros(alpha_shape, dtype=tf.float32)

        return calculate_h0, calculate_alphas

    def decoder(self, feature_grid, image_masks, input_characters, init_h, init_alphas,
                summarize=False, input_images=None):
        with tf.name_scope("decoder"):
            embedding = tf.get_variable(name="embedding", initializer=tf.initializers.random_normal,
                                        dtype=t.my_tf_float, shape=[self.vocabulary_size, self.embedding_dim])
            embedded_characters = tf.nn.embedding_lookup(embedding, input_characters)
            outputs, states = self._decoder(feature_grid=feature_grid, image_masks=image_masks,
                                            inputs=embedded_characters, init_h=init_h, init_alphas=init_alphas,
                                            summarize=summarize, input_images=input_images)

            dense_kernel = tf.get_variable(name="kernel_output_dense", dtype=t.my_tf_float,
                                            shape=[self.decoder_units, self.vocabulary_size - 1])
            output = tf.einsum("btf,fv->btv", outputs, dense_kernel, name="output")

        return output, states
