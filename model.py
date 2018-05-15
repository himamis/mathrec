from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, Concatenate, MaxPooling2D, BatchNormalization, Activation, Bidirectional, RNN, Embedding, LSTM, Reshape, RepeatVector, Lambda
from AttentionDecoderLSTM import AttentionDecoderLSTMCell

def create(vocabulary_size, embedding_size, encoder_size, imgH, imgW, free_run=False):

    imgs = Input(shape=(imgH, imgW, 1), dtype='float32', name='images') # (batch_size, imgH, imgW, 1)
    seqs = Input(shape=(None,), dtype='float32', name='sequences') # (batch_size, seq_len)  - seq_len = max of all lens of all sequences in this batch, where the additional length is filled with zeroes

    x = Lambda(lambda a: (a-128)/128)(imgs) # (batch_size, imgH, imgW, 1) - normilize to [-1, +1)

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

    # start row encoders
    rows = Lambda(lambda x: [x[:, i] for i in range(imgH>>3)])(x) # creates a list of tensors (batch_size, W, D) of len H
    row_encoders = []
    for i in range(len(rows)):
        row_encoders.append(Bidirectional(LSTM(encoder_size, return_sequences=True), merge_mode='concat')(rows[i])) # one row encoder: (batch_size, W, 2 * encoder_size);
    x = Concatenate(axis=1)(row_encoders) # (batch_size, H * W, 2 * encoder_size)
    # end row encoders

    cell = AttentionDecoderLSTMCell(vocabulary_size, (imgH>>3) * (imgW>>3), encoder_size * 2, embedding_size, free_run)
    y = Embedding(vocabulary_size, embedding_size)(seqs) # (batch_size, seq_len, embedding_size)
    y = RNN(cell, return_sequences=True)(y, constants=[x]) # (batch_size, seq_len, vocabulary_size)

    model = Model(inputs=(imgs, seqs), outputs=y)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model