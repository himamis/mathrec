from keras.models import Model
from keras.layers import Input, RNN, Conv2D, MaxPooling2D, BatchNormalization, Activation, \
    Bidirectional, LSTM, Lambda, Dense, Reshape
# if you use sometimes a current keras implementation, you don't need RNN and Reshape anymore and you can use it from keras
from trainer import AttentionDecoderLSTMCell
from trainer.defaults import create_vocabulary
from trainer.metrics import *
from keras.regularizers import l1, l1_l2, l2


def row_encoder(encoder_size, kernel_init, bias_init, name, x):
    # row encoder
    row = Bidirectional(LSTM(encoder_size, return_sequences=True, name=name, kernel_initializer=kernel_init,
                             bias_initializer=bias_init), merge_mode='concat')

    def step_foo(input_t, state):  # input_t: (batch_size, W, D), state doesn't matter
        return row(input_t), state  # (batch_size, W, 2 * encoder_size) 2 times encoder_size because of BiLSTM and concat

    l = Lambda(lambda x: K.rnn(step_foo, x, [])[1])(x)  # (batch_size, H, W, 2 * encoder_size)
    e = Reshape((-1, 2 * encoder_size))(l)

    return e


def create(vocabulary_size, encoder_size, internal_embedding=512, mask=None):
    # Weight initializers
    kernel_init = 'glorot_normal'
    bias_init = 'zeros'

    encoder_input_imgs = Input(shape=(None, None, 1), dtype='float32', name='encoder_input_images')  # (batch_size, imgH, imgW, 1)
    decoder_input = Input(shape=(None, vocabulary_size), dtype='float32', name='decoder_input_sequences')  # (batch_size, seq_len)

    # always use lambda if you want to change the tensor, otherwise you get a keras excption
    x = Lambda(lambda a: (a - 128) / 128)(encoder_input_imgs)  # (batch_size, imgH, imgW, 1) - normalize to [-1, +1)
    
    filter_sizes = [32, 64, 128, 256, 512]

    scales = []
    for filter_size in filter_sizes:
        # conv net
        x = Conv2D(filters=filter_size, kernel_size=3, strides=1, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)  # (batch_size, imgH, imgW, 64)
        x = Activation('relu')(x)
        x = Conv2D(filters=filter_size, kernel_size=3, strides=1, padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)  # (batch_size, imgH, imgW, 64)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=2, strides=2, padding='valid')(x)
        scales.append(x)


    encoder_large = row_encoder(256, kernel_init, bias_init, "encoder_large", scales[len(scales) - 1])
    encoder_small = row_encoder(128, kernel_init, bias_init, "encoder_small", scales[len(scales) - 2])

    # decoder
    regularization = None
    cell = AttentionDecoderLSTMCell(V=vocabulary_size, D=encoder_size * 2, D2= encoder_size, E=internal_embedding, regularizers=regularization)
    decoder = RNN(cell, return_sequences=True, return_state=True, name="decoder")
    decoder_output, _, _ = decoder(decoder_input, constants=[encoder_large, encoder_small])  # (batch_size, seq_len, encoder_size*2)
    decoder_dense = Dense(vocabulary_size, activation="softmax", kernel_initializer=kernel_init, bias_initializer=bias_init)
    decoder_output = decoder_dense(decoder_output)

    metrics = ['accuracy']
    if mask is not None:
        masked = get_masked_categorical_accuracy(mask)
        metrics.append(masked)

    model = Model(inputs=[encoder_input_imgs, decoder_input], outputs=decoder_output)
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=metrics)

    encoder_model = Model(encoder_input_imgs, [encoder_large, encoder_small])

    feature_grid_input = Input(shape=(None, 2 * encoder_size), dtype='float32', name='feature_grid')
    feature_grid_input_2 = Input(shape=(None, encoder_size), dtype='float32', name='feature_grid_2')
    decoder_state_h = Input(shape=(encoder_size * 2,))
    decoder_state_c = Input(shape=(encoder_size * 2,))
    decoder_output, state_h, state_c = decoder(decoder_input, constants=[feature_grid_input, feature_grid_input_2],
                                               initial_state=[decoder_state_h, decoder_state_c])
    decoder_output = decoder_dense(decoder_output)
    decoder_model = Model([feature_grid_input, feature_grid_input_2, decoder_input, decoder_state_h, decoder_state_c], [decoder_output, state_h, state_c])

    return model, encoder_model, decoder_model


def create_default(vocabulary_size=len(create_vocabulary()), mask=None):
    encoder_size = 256
    internal_embedding = 512
    return create(vocabulary_size, encoder_size, internal_embedding, mask)
