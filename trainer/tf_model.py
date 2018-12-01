from keras.models import Model
from keras.layers import Input, RNN, Conv2D, MaxPooling2D, BatchNormalization, Activation, \
    Bidirectional, LSTM, Lambda, Dense, Reshape
# if you use sometimes a current keras implementation, you don't need RNN and Reshape anymore and you can use it from keras
from trainer import AttentionDecoderLSTMCell
from trainer.defaults import create_vocabulary
from trainer.metrics import *
from keras.regularizers import l1, l1_l2, l2
from trainer.optimizer import PrintAdadelta
from keras.initializers import Orthogonal

import tensorflow as tf

def _row_encoder(feature_grid,
                 encoder_size=512,
                 rnn_kernel_init='orthogonal',
                 bidirectional=True):
    encoder = tf.keras.layers.LSTM(encoder_size, kernel_initializer=rnn_kernel_init,
                                   return_sequences=True, name="row_encoder_lstm")
    if bidirectional:
        encoder = Bidirectional(encoder)

    def step(x, _):
        output = encoder(x)
        return output, []

    _, outputs, _ = K.rnn(step, feature_grid, [])

    return outputs

def _decoder(vocabulary_size, hidden_size, feature_grid,
             lstm_kernel_initializer=None, lstm_bias_initializer=None):
    kernel = tf.get_variable(name="decoder_lstm_kernel_{}".format(hidden_size),
                             initializer=lstm_kernel_initializer,
                             shape=[vocabulary_size + hidden_size, 4 * hidden_size])
    bias = tf.get_variable(name="decoder_lstm_bias_{}".format(hidden_size),
                             initializer=lstm_bias_initializer,
                             shape=[4 * hidden_size])



    sigmoid = math_ops.sigmoid
    one = constant_op.constant(1, dtype=dtypes.int32)
    # Parameters of gates are concatenated into one multiply for efficiency.
    if self._state_is_tuple:
        c, h = state
    else:
        c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)

    gate_inputs = math_ops.matmul(
        array_ops.concat([inputs, h], 1), self._kernel)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, j, f, o = array_ops.split(
        value=gate_inputs, num_or_size_splits=4, axis=one)

    forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
    # Note that using `add` and `multiply` instead of `+` and `*` gives a
    # performance improvement. So using those at the cost of readability.
    add = math_ops.add
    multiply = math_ops.multiply
    new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),
                multiply(sigmoid(i), self._activation(j)))
    new_h = multiply(self._activation(new_c), sigmoid(o))

    if self._state_is_tuple:
        new_state = LSTMStateTuple(new_c, new_h)
    else:
        new_state = array_ops.concat([new_c, new_h], 1)
    return new_h, new_state

def _create(vocabulary_size, internal_embedding=512, mask=None,
           filter_sizes = [64, 128, 256, 512],
           conv_kernel_init='he_normal',
           conv_bias_init='zeros',
           conv_activation='relu',
           multi_scale_attention = False,
           is_training=True,
           batch_size=None, image_width=None, image_height=None):
    encoder_input_imgs = tf.placeholder(tf.float32, shape=(batch_size, image_height, image_width, 1), name="input_image")
    conv = (encoder_input_imgs - 128) / 128
    prev_conv = None
    for index, filter_size in enumerate(filter_sizes):
        prev_conv = conv
        conv = tf.layers.conv2d(conv, filters=filter_size, kernel_size=3, strides=1, padding='same',
                                kernel_initializer=conv_kernel_init, bias_initializer=conv_bias_init,
                                activation=conv_activation,
                                name="conv_block_{}_filter_{}_1".format(index, filter_size))
        conv = tf.layers.conv2d(conv, filters=filter_size, kernel_size=3, strides=1, padding='same',
                                kernel_initializer=conv_kernel_init, bias_initializer=conv_bias_init,
                                activation=None,
                                name="conv_block_{}_filter_{}_2".format(index, filter_size))
        conv = tf.layers.batch_normalization(conv, training=is_training,
                                             name="batch_norm_block_{}_filter_{}".format(index, filter_size))
        conv = tf.nn.relu(conv, name="relu_block_{}_filter_{}".format(index, filter_size))
        conv = tf.layers.max_pooling2d(conv, pool_size=2, strides=2, padding='valid',
                                       name="max_pooling_{}_filter_{}".format(index, filter_size))

    encoded = _row_encoder(conv, filter_size[-1])
    encoded_larger_scale = None
    if multi_scale_attention:
        encoded_larger_scale = _row_encoder(prev_conv, filter_size[-2])



    return encoded


_create(70, 48)

def row_encoder(encoder_size, kernel_init, bias_init, name, x):
    # row encoder
    row = Bidirectional(LSTM(encoder_size, return_sequences=True, name=name, kernel_initializer=kernel_init,
                             bias_initializer=bias_init), merge_mode='concat')

    # row = LSTM(encoder_size, return_sequences=True, name=name, kernel_initializer=kernel_init, bias_initializer=bias_init)

    def step_foo(input_t, state):  # input_t: (batch_size, W, D), state doesn't matter
        return row(
            input_t), state  # (batch_size, W, 2 * encoder_size) 2 times encoder_size because of BiLSTM and concat

    l = Lambda(lambda x: K.rnn(step_foo, x, [])[1])(x)  # (batch_size, H, W, 2 * encoder_size)
    e = Reshape((-1, 2 * encoder_size))(l)
    # e = Reshape((-1, encoder_size))(l)

    return e


def create(vocabulary_size, encoder_size, internal_embedding=512, mask=None):
    # Weight initializers
    rnn_kernel_init = Orthogonal()
    cnn_kernel_init = 'he_normal'
    dense_init = 'glorot_normal'
    bias_init = 'zeros'

    encoder_input_imgs = Input(shape=(None, None, 1), dtype='float32',
                               name='encoder_input_images')  # (batch_size, imgH, imgW, 1)
    decoder_input = Input(shape=(None, vocabulary_size), dtype='float32',
                          name='decoder_input_sequences')  # (batch_size, seq_len)

    # always use lambda if you want to change the tensor, otherwise you get a keras excption
    x = Lambda(lambda a: (a - 128) / 128)(encoder_input_imgs)  # (batch_size, imgH, imgW, 1) - normalize to [-1, +1)

    filter_sizes = [64, 128, 256, 512]

    scales = []
    for filter_size in filter_sizes:
        # conv net
        x = Conv2D(filters=filter_size, kernel_size=3, strides=1, padding='same', kernel_initializer=cnn_kernel_init,
                   bias_initializer=bias_init)(x)  # (batch_size, imgH, imgW, 64)
        x = Activation('relu')(x)
        x = Conv2D(filters=filter_size, kernel_size=3, strides=1, padding='same', kernel_initializer=cnn_kernel_init,
                   bias_initializer=bias_init)(x)  # (batch_size, imgH, imgW, 64)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=2, strides=2, padding='valid')(x)
        scales.append(x)

    encoder_large = row_encoder(encoder_size, rnn_kernel_init, bias_init, "encoder_large", scales[len(scales) - 1])
    encoder_small = row_encoder(int(encoder_size / 2), rnn_kernel_init, bias_init, "encoder_small",
                                scales[len(scales) - 2])

    # decoder
    regularization = None
    cell = AttentionDecoderLSTMCell(V=vocabulary_size, D=encoder_size * 2, D2=encoder_size, E=internal_embedding,
                                    regularizers=regularization, dense_initializer=dense_init,
                                    kernel_initializer=rnn_kernel_init, bias_initializer=bias_init)
    # cell = AttentionDecoderLSTMCell(V=vocabulary_size, D=encoder_size, D2=int(encoder_size/2), E=internal_embedding, regularizers=regularization,dense_initializer=dense_init,kernel_initializer=rnn_kernel_init,bias_initializer=bias_init)
    decoder = RNN(cell, return_sequences=True, return_state=True, name="decoder")
    decoder_output, _, _ = decoder(decoder_input,
                                   constants=[encoder_large, encoder_small])  # (batch_size, seq_len, encoder_size*2)
    # decoder_output, _, _ = decoder(decoder_input, constants=[encoder_large])  # (batch_size, seq_len, encoder_size*2)
    decoder_dense = Dense(vocabulary_size, activation="softmax", kernel_initializer=dense_init,
                          bias_initializer=bias_init)
    decoder_output = decoder_dense(decoder_output)

    metrics = ['accuracy']
    if mask is not None:
        masked = get_masked_categorical_accuracy(mask)
        metrics.append(masked)

    model = Model(inputs=[encoder_input_imgs, decoder_input], outputs=decoder_output)
    model.compile(optimizer=PrintAdadelta(), loss='categorical_crossentropy', metrics=metrics)

    encoder_model = Model(encoder_input_imgs, [encoder_large, encoder_small])
    # encoder_model = Model(encoder_input_imgs, [encoder_large])

    feature_grid_input = Input(shape=(None, 2 * encoder_size), dtype='float32', name='feature_grid')
    feature_grid_input_2 = Input(shape=(None, encoder_size), dtype='float32', name='feature_grid_2')
    # feature_grid_input = Input(shape=(None, encoder_size), dtype='float32', name='feature_grid')
    # feature_grid_input_2 = Input(shape=(None, int(encoder_size/2)), dtype='float32', name='feature_grid_2')
    decoder_state_h = Input(shape=(encoder_size * 2,))
    decoder_state_c = Input(shape=(encoder_size * 2,))
    # decoder_state_h = Input(shape=(encoder_size,))
    # decoder_state_c = Input(shape=(encoder_size,))

    decoder_output, state_h, state_c = decoder(decoder_input, constants=[feature_grid_input, feature_grid_input_2],
                                               initial_state=[decoder_state_h, decoder_state_c])
    # decoder_output, state_h, state_c = decoder(decoder_input, constants=[feature_grid_input],
    #                                           initial_state=[decoder_state_h, decoder_state_c])
    decoder_output = decoder_dense(decoder_output)
    # decoder_model = Model([feature_grid_input, decoder_input, decoder_state_h, decoder_state_c], [decoder_output, state_h, state_c])
    decoder_model = Model([feature_grid_input, feature_grid_input_2, decoder_input, decoder_state_h, decoder_state_c],
                          [decoder_output, state_h, state_c])

    return model, encoder_model, decoder_model


def create_default(vocabulary_size=len(create_vocabulary()), mask=None):
    encoder_size = 256
    internal_embedding = 512
    return create(vocabulary_size, encoder_size, internal_embedding, mask)
