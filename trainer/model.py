from keras import backend as K
from keras.models import Model
from keras.layers import Input, RNN, Conv2D, Concatenate, MaxPooling2D, BatchNormalization, Activation, Bidirectional, Embedding, LSTM, Lambda, Flatten
# if you use sometimes a current keras implementation, you don't need RNN and Reshape anymore and you can use it from keras
from trainer import AttentionDecoderLSTMCell, Reshape
from trainer.defaults import create_vocabulary

def create(vocabulary_size, embedding_size, encoder_size, free_run=False):

    imgs = Input(shape=(None, None, 3), dtype='float32', name='images') # (batch_size, imgH, imgW, 1)
    seqs = Input(shape=(None, 1), dtype='float32', name='sequences') # (batch_size, seq_len)
    
    # always use lambda if you want to change the tensor, otherwise you get a keras excption
    x = Lambda(lambda a: (a-128)/128)(imgs) # (batch_size, imgH, imgW, 3) - normalize to [-1, +1)

    # conv net
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(x) # (batch_size, imgH, imgW, 64)
    x = MaxPooling2D(pool_size=2, strides=2, padding='valid')(x) # (batch_size, imgH/2, imgW/2, 64)

    x = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(x) # (batch_size, imgH/2, imgW/2, 128)
    x = MaxPooling2D(pool_size=2, strides=2, padding='valid')(x) # (batch_size, imgH/2/2, imgW/2/2, 128)

    x = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(x) # (batch_size, imgH/2/2, imgW/2/2,  256)
    x = BatchNormalization(scale=False)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(x) # (batch_size, imgH/2/2, imgW/2/2, 256)
    x = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid')(x) # (batch_size, imgH/2/2/2, imgW/2/2, 256)

    x = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(x) # (batch_size, imgH/2/2/2, imgW/2/2, 512)
    x = BatchNormalization(scale=False)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='valid')(x) # (batch_size, imgH/2/2/2, imgW/2/2/2, 512)

    x = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(x) # (batch_size, imgH/2/2/2, imgW/2/2/2, 512) = (batch_size, H, W, D)
    x = BatchNormalization(scale=False)(x)
    x = Activation('relu')(x)

    # row encoder
    row = Bidirectional(LSTM(encoder_size, return_sequences=True), merge_mode='concat')

    def step_foo(input_t, state): # input_t: (batch_size, W, D), state doesn't matter
        return row(input_t), state # (batch_size, W, 2 * encoder_size) 2 times encoder_size because of BiLSTM and concat
    # Important to use a lambda outside a layer
    x = Lambda(lambda x: K.rnn(step_foo, x, [])[1])(x) # (batch_size, H, W, 2 * encoder_size)
    encoder = Reshape((-1, 2 * encoder_size))(x) # (batch_size, H * W, 2 * encoder_size) H * W = L in AttentionDecoderLSTMCell

    # decoder
    cell = AttentionDecoderLSTMCell(vocabulary_size, encoder_size * 2, embedding_size, free_run)
    decoder = RNN(cell, return_sequences=True)
    y = decoder(seqs, constants=[encoder])  # (batch_size, seq_len, vocabulary_size)

    model = Model(inputs=[imgs, seqs], outputs=y)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model, encoder, decoder


def create_default(vocabulary_size=len(create_vocabulary()), free_run=False):
    embedding_size = 80  # not needed in current version
    encoder_size = 256
    return create(vocabulary_size, embedding_size, encoder_size, free_run)
