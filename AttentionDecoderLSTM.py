from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Embedding
import numpy as np


class AttentionDecoderLSTMCell(Layer):

    def __init__(self, V = 0, L = 0, D = 0, E = 0, free_run=True, **kwargs):
        self.L = L
        self.D = D
        self.E = E
        self.V = V
        self.state_size = (V, self.D, self.D, self.D) # (y, out, h, c)
        self.free_run = free_run
        self.__embedding_layer = Embedding(V, E)
        super(AttentionDecoderLSTMCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_f = self.add_weight(name='W_f', shape=(self.D + self.E, self.D), initializer='uniform', trainable=True)
        self.W_g = self.add_weight(name='W_g', shape=(self.D + self.E, self.D), initializer='uniform', trainable=True)
        self.W_i = self.add_weight(name='W_i', shape=(self.D + self.E, self.D), initializer='uniform', trainable=True)
        self.W_o = self.add_weight(name='W_o', shape=(self.D + self.E, self.D), initializer='uniform', trainable=True)
        self.b_f = self.add_weight(name='b_f', shape=(self.D,), initializer='uniform', trainable=True)
        self.b_g = self.add_weight(name='b_g', shape=(self.D,), initializer='uniform', trainable=True)
        self.b_i = self.add_weight(name='b_i', shape=(self.D,), initializer='uniform', trainable=True)
        self.b_o = self.add_weight(name='b_o', shape=(self.D,), initializer='uniform', trainable=True)

        self.W_e = self.add_weight(name='W_e', shape=(self.D, self.D), initializer='uniform', trainable=True)
        
        self.W_out = self.add_weight(name='W_out', shape=(2*self.D, self.D), initializer='uniform', trainable=True)
        
        self.W_y = self.add_weight(name='W_y', shape=(self.D, self.V), initializer='uniform', trainable=True)

        self.built = True

    def call(self, inputs, states, constants=None):
        featureGrid = constants[0]

        # Embedding
        if self.free_run:
            _input = K.argmax(states[0]) # (batch_size)
            _input = self.__embedding_layer(_input) # (batch_size, E)
        else:
            _input = inputs # (batch_size, E)
        x = K.concatenate((_input, states[1])) # (batch_size, E + D)

        # LSTM
        x = K.expand_dims(x, 1) # (batch_size, 1, E + D)
        f = K.sigmoid(K.dot(x, self.W_f) + self.b_f)[:,0] # (batch_size, D)
        i = K.sigmoid(K.dot(x, self.W_i) + self.b_i)[:,0] # (batch_size, D)
        g = K.tanh(K.dot(x, self.W_g) + self.b_g)[:,0] # (batch_size, D)
        o = K.sigmoid(K.dot(x, self.W_o) + self.b_o)[:,0] # (batch_size, D)

        c = states[3] * f + i * g # (batch_size, D)
        h = K.tanh(c) * o # (batch_size, D)

        # Attention
        b = K.dot(K.expand_dims(h, 1), self.W_e)[:,0] # (batch_size, D)
        b = K.expand_dims(b) # (batch_size, D, 1)
        e = K.batch_dot(featureGrid, b)[:,:,0] # (batch_size, L)
        a = K.softmax(e) # (batch_size, L)
        a = K.expand_dims(a, 1) # (batch_size, 1, L)
        z = K.batch_dot(a, featureGrid)[:,0] # (batch_size, D)
        
        # Output
        hz = K.concatenate((h, z)) # (batch_size, 2D,)
        hz = K.expand_dims(hz, 1) # (batch_size, 1, 2D)
        out = K.tanh(K.dot(hz, self.W_out)) # (batch_size, 1, D)
        y = K.softmax(K.dot(out, self.W_y)[:,0]) # (batch_size, V)
        out = out[:,0] # (batch_size, D)

        return y, [y, out, h, c]

    def get_config(self):
        config = super().get_config()
        config['V'] = self.V
        config['E'] = self.E
        config['L'] = self.L
        config['D'] = self.D
        config['free_run'] = self.free_run
        return config

    def from_config(cls, config):
        cls.L = config['L']
        cls.D = config['D']
        cls.E = config['E']
        cls.V = config['V']
        cls.free_run = config['free_run']
        cls.state_size = (cls.V, cls.D, cls.D, cls.D) # (y, out, h, c)
        cls.__embedding_layer = Embedding(cls.V, cls.E)
        return cls