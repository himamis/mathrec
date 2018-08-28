from keras import backend as K
from keras.models import Model
from keras.layers import Input, RNN, Conv2D, MaxPooling2D, BatchNormalization, Activation, \
    Bidirectional, LSTM, Lambda, Dense, Reshape
from keras.layers import Embedding
# if you use sometimes a current keras implementation, you don't need RNN and Reshape anymore and you can use it from keras
from trainer import AttentionLSTMDecoderCell
from trainer.defaults import create_vocabulary


def create(vocabulary_size, embedding_size, encoder_size):
    imgs = Input(shape=(256, 512, 3), dtype='float32', name='images')  # (batch_size, imgH, imgW, 1)
    seqs = Input(shape=(None, ), dtype='float32', name='sequences')  # (batch_size, seq_len)

    # always use lambda if you want to change the tensor, otherwise you get a keras excption
    x = Lambda(lambda a: (a - 128) / 128)(imgs)  # (batch_size, imgH, imgW, 3) - normalize to [-1, +1)

    # conv net
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(
        x)  # (batch_size, imgH, imgW, 64)
    x = MaxPooling2D(pool_size=2, strides=2, padding='valid')(x)  # (batch_size, imgH/2, imgW/2, 64)

    x = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(
        x)  # (batch_size, imgH/2, imgW/2, 128)
    x = MaxPooling2D(pool_size=2, strides=2, padding='valid')(x)  # (batch_size, imgH/2/2, imgW/2/2, 128)

    x = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(x)  # (batch_size, imgH/2/2, imgW/2/2,  256)
    x = BatchNormalization(scale=False)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(
        x)  # (batch_size, imgH/2/2, imgW/2/2, 256)
    x = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid')(x)  # (batch_size, imgH/2/2/2, imgW/2/2, 256)

    x = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(x)  # (batch_size, imgH/2/2/2, imgW/2/2, 512)
    x = BatchNormalization(scale=False)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='valid')(x)  # (batch_size, imgH/2/2/2, imgW/2/2/2, 512)

    x = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(
        x)  # (batch_size, imgH/2/2/2, imgW/2/2/2, 512) = (batch_size, H, W, D)
    x = BatchNormalization(scale=False)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid')(x)
    # (batch_size, 32, 32, 512)

    # row encoder
    row = Bidirectional(LSTM(encoder_size, return_sequences=True, name="encoder"), merge_mode='concat')

    def step_foo(input_t, state):  # input_t: (batch_size, W, D), state doesn't matter
        return row(input_t), state  # (batch_size, W, 2 * encoder_size) 2 times encoder_size because of BiLSTM and concat

    # Important to use a lambda outside a layer
    x = Lambda(lambda x: K.rnn(step_foo, x, [])[1])(x)  # (batch_size, H, W, 2 * encoder_size)
    encoder = Reshape((-1, 2 * encoder_size))(x)  # (batch_size, H * W, 2 * encoder_size) H * W = L in AttentionDecoderLSTMCell

    image_average = Lambda(lambda x: K.mean(x, axis=1))
    average_encoder_feature = image_average(encoder)
    initial_state_h_dense = Dense(vocabulary_size)
    initial_state_c_dense = Dense(vocabulary_size)
    initial_state_h = initial_state_h_dense(average_encoder_feature)
    initial_state_c = initial_state_c_dense(average_encoder_feature)

    embedding = Embedding(vocabulary_size, embedding_size)(seqs)

    # decoder
    cell = AttentionLSTMDecoderCell(vocabulary_size, encoder_size * 2, 200)
    decoder = RNN(cell, return_sequences=True, name="decoder")
    y = decoder(embedding, constants=[encoder], initial_state=[initial_state_h, initial_state_c])  # (batch_size, seq_len, vocabulary_size)

    model = Model(inputs=[imgs, seqs], outputs=y)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    encoder_model = Model(imgs, encoder)

    feature_grid_input = Input(shape=(32 * 32, 512), dtype='float32', name='feature_grid')
    average_input = image_average(feature_grid_input)
    initial_state_h = initial_state_h_dense(average_input)
    initial_state_c = initial_state_c_dense(average_input)
    output = decoder(embedding, constants=[feature_grid_input], initial_state=[initial_state_h, initial_state_c])

    decoder_model = Model(inputs=[seqs, feature_grid_input], outputs=output)
    return model, encoder_model, decoder_model


def create_default(vocabulary_size=len(create_vocabulary())):
    embedding_size = 80  # not needed in current version
    encoder_size = 256
    return create(vocabulary_size, embedding_size, encoder_size)
