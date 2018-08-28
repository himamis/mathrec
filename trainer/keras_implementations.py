from keras import backend as K
from keras.layers import Embedding, LSTMCell, Wrapper
import numpy as np
import file_utils as utils
import warnings
from keras.callbacks import Callback
import traceback
from numpy import random
import sys
import inspect
from keras.engine.topology import Layer
from keras.engine.topology import InputSpec


class AttentionLSTMDecoderCell(LSTMCell):

    def __init__(self, output_dim, feature_size, alignment_hidden,  **kwargs):
        self.output_dim = output_dim
        self.feature_size = feature_size
        self.alignment_hidden = alignment_hidden

        super(AttentionLSTMDecoderCell, self).__init__(output_dim, **kwargs)

    def build(self, input_shape):
        self.Ua = self.add_weight(name="Ua", shape=(self.feature_size, self.alignment_hidden), initializer='uniform')
        self.Ua_b = self.add_weight(name="Ua_b", shape=(self.alignment_hidden,), initializer='uniform')

        self.Wa = self.add_weight(name="Wa", shape=(self.output_dim, self.alignment_hidden), initializer='uniform')
        self.Wa_b = self.add_weight(name="Wa_b", shape=(self.alignment_hidden,), initializer='uniform')

        self.va = self.add_weight(name="va", shape=(self.alignment_hidden,), initializer='uniform')
        super(AttentionLSTMDecoderCell, self).build((None, input_shape[0][1] + input_shape[1][2] + self.output_dim))


    def call(self, inputs, states, training=None, constants=None):
        # Calculate context
        feature_grid = constants[0] # (batch_size, h*w, feature_size)
        s = states[0] # (batch_size, output_dim)
        Uahj = K.dot(feature_grid, self.Ua) + self.Ua_b # (batch_size, h*w, alignment_hidden)
        Wasi = K.dot(s, self.Wa) + self.Wa_b # (batch_size, alignment_hidden)
        sm = K.tanh(K.expand_dims(Wasi, 1) + Uahj) # (batch_size, h*w, alignment_hidden)
        e = K.dot(sm, K.expand_dims(self.va, 1))[:,:,0] # (batch_size, h*w)
        a = K.softmax(e) # (batch_size, h*w)
        c = K.batch_dot(K.expand_dims(a, 1), feature_grid)[:, 0, :] # (batch_size, feature_size)

        new_inputs = K.concatenate((inputs, states[0], c))

        h, [h, c] = super(AttentionLSTMDecoderCell, self).call(new_inputs, states, training=training)

        return h, [h, c]


class AttentionDecoderLSTMCell(Layer):
    '''
    An implementation of an LSTM cell that supports attention.

    Arguments:
        V - vocabulary size
        D - number of hidden states
        E - embedding size; currently not used
        free_run - False is normally training and evaluation and True is normally used for predicting. If True, 
        it uses the previous generated token to generate the next one. If False, it uses the inputs param in the 
        call method
    '''

    def __init__(self, V = 0, D = 0, E = 0, free_run=True, **kwargs):
        self.D = D
        self.E = E
        self.V = V
        self.state_size = (V, self.D, self.D, self.D) # (y, out, h, c)
        self.free_run = free_run
        super(AttentionDecoderLSTMCell, self).__init__(**kwargs)

    def build(self, input_shape):
        # generating weights. In future one maybe implements biases for W_e, W_out, W_y...
        self.W_f = self.add_weight(name='W_f', shape=(self.D + self.V, self.D), initializer='uniform', trainable=True)
        self.W_g = self.add_weight(name='W_g', shape=(self.D + self.V, self.D), initializer='uniform', trainable=True)
        self.W_i = self.add_weight(name='W_i', shape=(self.D + self.V, self.D), initializer='uniform', trainable=True)
        self.W_o = self.add_weight(name='W_o', shape=(self.D + self.V, self.D), initializer='uniform', trainable=True)
        self.b_f = self.add_weight(name='b_f', shape=(self.D,), initializer='uniform', trainable=True)
        self.b_g = self.add_weight(name='b_g', shape=(self.D,), initializer='uniform', trainable=True)
        self.b_i = self.add_weight(name='b_i', shape=(self.D,), initializer='uniform', trainable=True)
        self.b_o = self.add_weight(name='b_o', shape=(self.D,), initializer='uniform', trainable=True)
        self.W_e = self.add_weight(name='W_e', shape=(self.D, self.D), initializer='uniform', trainable=True)
        self.W_out = self.add_weight(name='W_out', shape=(2*self.D, self.D), initializer='uniform', trainable=True)
        self.W_y = self.add_weight(name='W_y', shape=(self.D, self.V), initializer='uniform', trainable=True)
        self.built = True # important!!
        # super([Layer], self).build()

    def call(self, inputs, states, constants=None):
        featureGrid = constants[0]
        # Input
        if self.free_run:
            _input = K.argmax(states[0]) # (batch_size)
        else:
            _input = inputs
            #_input = K.cast(inputs[:, 0], 'int32') # (batch_size)
        #_input = K.one_hot(_input, self.V) # (batch_size, V)
        x = K.concatenate((_input, states[1])) # (batch_size, V + D)

        # LSTM
        x = K.expand_dims(x, 1) # (batch_size, 1, V + D)
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

    # if you want sometimes to save the model architecture
    def get_config(self):
        config = super().get_config()
        config['V'] = self.V
        config['E'] = self.E
        config['D'] = self.D
        config['free_run'] = self.free_run
        return config

    # if you want sometimes to load the model architecture
    def from_config(cls, config):
        cls.D = config['D']
        cls.E = config['E']
        cls.V = config['V']
        cls.free_run = config['free_run']
        cls.state_size = (cls.V, cls.D, cls.D, cls.D) # (y, out, h, c)
        return cls


class AttentionLSTMCell(Layer):
    '''
    Another implementation of an LSTM cell that supports attention.

    Arguments:
        V - vocabulary size
        D - number of hidden states
        E - embedding size; currently not used
        free_run - False is normally training and evaluation and True is normally used for predicting. If True, 
        it uses the previous generated token to generate the next one. If False, it uses the inputs param in the 
        call method
    '''

    def __init__(self, V = 0, D = 0, E = 0, free_run=True, **kwargs):
        self.D = D
        self.E = E
        self.V = V
        self.state_size = (V, self.D, self.D, self.D) # (y, out, h, c)
        self.free_run = free_run
        self.__embedding_layer = Embedding(V, E)
        super(AttentionLSTMCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_f = self.add_weight(name='W_f', shape=(2 * self.D + self.V, self.D), initializer='uniform', trainable=True)
        self.W_g = self.add_weight(name='W_g', shape=(2 * self.D + self.V, self.D), initializer='uniform', trainable=True)
        self.W_i = self.add_weight(name='W_i', shape=(2 * self.D + self.V, self.D), initializer='uniform', trainable=True)
        self.W_o = self.add_weight(name='W_o', shape=(2 * self.D + self.V, self.D), initializer='uniform', trainable=True)
        self.b_f = self.add_weight(name='b_f', shape=(self.D,), initializer='uniform', trainable=True)
        self.b_g = self.add_weight(name='b_g', shape=(self.D,), initializer='uniform', trainable=True)
        self.b_i = self.add_weight(name='b_i', shape=(self.D,), initializer='uniform', trainable=True)
        self.b_o = self.add_weight(name='b_o', shape=(self.D,), initializer='uniform', trainable=True)
        self.W_e = self.add_weight(name='W_e', shape=(self.D, self.D), initializer='uniform', trainable=True)
        self.W_out = self.add_weight(name='W_out', shape=(2*self.D, self.D), initializer='uniform', trainable=True)
        self.W_y = self.add_weight(name='W_y', shape=(self.D, self.V), initializer='uniform', trainable=True)
        self.built = True

    def call(self, inputs, states, constants=None):
        featureGrid = constants[0] # (batch_size, L, D)
        htm1 = states[1] # (batch_size, D)
        ytm1 = states[0] # (batch_size, V)

        # Attention
        b = K.dot(K.expand_dims(htm1, 1), self.W_e)[:, 0] # (batch_size, D)
        b = K.expand_dims(b) # (batch_size, D, 1)
        e = K.batch_dot(featureGrid, b)[:, :, 0] # (batch_size, L)
        a = K.softmax(e) # (batch_size, L)
        a = K.expand_dims(a, 1) # (batch_size, 1, L)
        c = K.batch_dot(a, featureGrid)[:, 0] # (batch_size, D)
        
        # Input
        if self.free_run:
            _input = K.argmax(states[0]) # (batch_size)
        else:
            _input = K.cast(inputs[:, 0], 'int32') # (batch_size)
        _input = K.one_hot(_input, self.V) # (batch_size, V)
        
        # LSTM
        x = K.concatenate((_input, htm1, c)) # (batch_size, V + 2D)
        x = K.expand_dims(x, 1) # (batch_size, 1, V + 2D)
        f = K.sigmoid(K.dot(x, self.W_f) + self.b_f)[:,0] # (batch_size, D)
        i = K.sigmoid(K.dot(x, self.W_i) + self.b_i)[:,0] # (batch_size, D)
        g = K.tanh(K.dot(x, self.W_g) + self.b_g)[:,0] # (batch_size, D)
        o = K.sigmoid(K.dot(x, self.W_o) + self.b_o)[:,0] # (batch_size, D)
        h = htm1 * f + i * g # (batch_size, D)
        out = K.tanh(h) * o # (batch_size, D)
        out = K.expand_dims(out, 1) # (batch_size, 1, D)
        y = K.dot(out, self.W_y)[:, 0] # (batch_size, V)
        y = K.softmax(y) # (batch_size, V)

        return y, [y, h]

    def get_config(self):
        config = super().get_config()
        config['V'] = self.V
        config['E'] = self.E
        config['D'] = self.D
        config['free_run'] = self.free_run
        return config

    def from_config(cls, config):
        cls.D = config['D']
        cls.E = config['E']
        cls.V = config['V']
        cls.free_run = config['free_run']
        cls.state_size = (cls.V, cls.D, cls.D, cls.D) # (y, out, h, c)
        return cls


# Nearly the same implementation as ModelCheckpoint from the keras team. 
# The reason for this own implementation is that the original one couldnt't
# save to or load from Google Cloud Storage.
class ModelCheckpointer(Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
    """
    
    def __init__(self, filepath, monitor='val_loss', verbose=0, mode='auto', period=1):
        super(ModelCheckpointer, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpointer mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            current = logs.get(self.monitor)
            if current is not None:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f' % (epoch + 1, self.monitor, self.best, current))
                    self.best = current
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s did not improve from %0.5f' % (epoch + 1, self.monitor, self.best))
            if self.verbose > 0:
                print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
            weights = self.model.get_weights()
            utils.write_npy(filepath, weights)


class SequenceGenerator:
    """
    A generator, which can be used for the Model class of Keras for the methods fit_generator(...) and
    evaluate_generator(...).

    set_dir - path to file: gs://<bucket-name>/.../
    img_dir - path to images: normally set_dir/images/
    set - either train, validate or test
    batch_size - size of batches
    """

    def __init__(self, set_file, img_dir, batch_size=32):
        # list: [(img_name, seq, size),...], img_name: str, seq: [token,...], token: int, size: (width, height) of image
        self.set_list = utils.read_pkl(set_file) 
        self.img_dir = img_dir
        self.batch_size = batch_size
         # TODO: remove -1 and make last batch in __getitem__ smaller if needed
        self.len = int(np.ceil(len(self.set_list) / float(self.batch_size))) - 1
        self.next_list = []

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        batch_x = self.set_list[i*self.batch_size:(i + 1) * self.batch_size]
        # it is ordered by length of seq
        max_len = len(batch_x[-1][1])
        # create array of size max_len with values one. One is the end-token <et>.
        # so if the first ones are shorter, they become filled up with end-tokens
        x_seqs = np.ones((self.batch_size, max_len), dtype=np.int32)
        x_imgs = []
        # find max width and height, because all images in one batch must have the same size
        max_width, max_height = 0, 0
        for name, seq, size in batch_x:
            max_height = max(max_height, size[1])
            max_width = max(max_width, size[0])

        for n, x in enumerate(batch_x):
            # x[0] = image name, x[1] = list of tokens
            # set tokens to x_seqs. Remember if this sequence is smaller, there are ones (end-tokens) at the end
            x_seqs[n][:len(x[1])] = x[1]
            # because there are sometimes errors loading images from google cloud storage, try it at least 5 times
            for j in range(5):
                try:
                    # reads image and resizes it to the max height and width. It also
                    # converts it to the YCbCr format. This method already returns a numpy array.
                    img = utils.read_img(self.img_dir+x[0], (max_width, max_height))
                    break
                except Exception as e:
                    print('Image2Latex: Could not load image', j, n, i)
                    print('Image2Latex:', e)
                    traceback.print_stack()
                    img = None
            if img is None:
                raise Exception('Image2Latex: unfähig', n, i)
            # before appending img has dimension [height,width,3(YCbCr)], after [height, width, 1]
            x_imgs.append(img[:, :, 0][:, :, None])
        # since the whole sequence looks like: <st> ... <et> we have to create two different
        # y_seqs: ... <et>, is here for comparing the results while training, validating and evaluating
        y_seqs = x_seqs[:,1:,None]
        # x_seqs: <st> ..., is here for none-free_run mode
        x_seqs = x_seqs[:,:-1,None]
        print('Image2Latex: rest batches', len(self.next_list))
        # don't forget that every input and output has to be a numpy array. But when grouping inputs or outputs
        # you have to use lists (also no tuples!)
        return [np.array(x_imgs), np.array(x_seqs)], y_seqs

    def __iter__(self):
        return self

    # for python 3
    def __next__(self):
        return self.next()

    # for python 2
    def next(self):
        # method is endless. Alwys takes randomly the next batch. If all batches are used, it starts again.
        if len(self.next_list) == 0:
            self.next_list = list(range(self.len))
        index = random.randint(0, len(self.next_list))
        cur = self.next_list[index]
        del self.next_list[index]
        return self[cur]


# Just a copy from current keras implementation, because in keras 2.0.4 reshaping None-dim was not able.
class Reshape(Layer):
    """Reshapes an output to a certain shape.
    # Arguments
        target_shape: target shape. Tuple of integers.
            Does not include the batch axis.
    # Input shape
        Arbitrary, although all dimensions in the input shaped must be fixed.
        Use the keyword argument `input_shape`
        (tuple of integers, does not include the batch axis)
        when using this layer as the first layer in a model.
    # Output shape
        `(batch_size,) + target_shape`
    # Example
    ```python
        # as first layer in a Sequential model
        model = Sequential()
        model.add(Reshape((3, 4), input_shape=(12,)))
        # now: model.output_shape == (None, 3, 4)
        # note: `None` is the batch dimension
        # as intermediate layer in a Sequential model
        model.add(Reshape((6, 2)))
        # now: model.output_shape == (None, 6, 2)
        # also supports shape inference using `-1` as dimension
        model.add(Reshape((-1, 2, 2)))
        # now: model.output_shape == (None, 3, 2, 2)
    ```
    """

    def __init__(self, target_shape, **kwargs):
        super(Reshape, self).__init__(**kwargs)
        self.target_shape = tuple(target_shape)

    def _fix_unknown_dimension(self, input_shape, output_shape):
        """Finds and replaces a missing dimension in an output shape.
        This is a near direct port of the internal Numpy function
        `_fix_unknown_dimension` in `numpy/core/src/multiarray/shape.c`
        # Arguments
            input_shape: original shape of array being reshaped
            output_shape: target shape of the array, with at most
                a single -1 which indicates a dimension that should be
                derived from the input shape.
        # Returns
            The new output shape with a `-1` replaced with its computed value.
        # Raises
            ValueError: if `input_shape` and `output_shape` do not match.
        """
        output_shape = list(output_shape)
        msg = 'total size of new array must be unchanged'

        known, unknown = 1, None
        for index, dim in enumerate(output_shape):
            if dim < 0:
                if unknown is None:
                    unknown = index
                else:
                    raise ValueError('Can only specify one unknown dimension.')
            else:
                known *= dim

        original = np.prod(input_shape, dtype=int)
        if unknown is not None:
            if known == 0 or original % known != 0:
                raise ValueError(msg)
            output_shape[unknown] = original // known
        elif original != known:
            raise ValueError(msg)

        return tuple(output_shape)

    def compute_output_shape(self, input_shape):
        if None in input_shape[1:]:
            # input shape (partially) unknown? replace -1's with None's
            return ((input_shape[0],) +
                    tuple(s if s != -1 else None for s in self.target_shape))
        else:
            # input shape known? then we can compute the output shape
            return (input_shape[0],) + self._fix_unknown_dimension(
                input_shape[1:], self.target_shape)

    def call(self, inputs):
        return K.reshape(inputs, (K.shape(inputs)[0],) + self.target_shape)

    def get_config(self):
        config = {'target_shape': self.target_shape}
        base_config = super(Reshape, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Same implementation as from keras. The reason is that in keras 2.0.4 it wasn't implmented already.
class RNN(Layer):
    """Base class for recurrent layers.
    # Arguments
        cell: A RNN cell instance. A RNN cell is a class that has:
            - a `call(input_at_t, states_at_t)` method, returning
                `(output_at_t, states_at_t_plus_1)`. The call method of the
                cell can also take the optional argument `constants`, see
                section "Note on passing external constants" below.
            - a `state_size` attribute. This can be a single integer
                (single state) in which case it is
                the size of the recurrent state
                (which should be the same as the size of the cell output).
                This can also be a list/tuple of integers
                (one size per state). In this case, the first entry
                (`state_size[0]`) should be the same as
                the size of the cell output.
            It is also possible for `cell` to be a list of RNN cell instances,
            in which cases the cells get stacked on after the other in the RNN,
            implementing an efficient stacked RNN.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively,
            the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
        input_length: Length of input sequences, to be specified
            when it is constant.
            This argument is required if you are going to connect
            `Flatten` then `Dense` layers upstream
            (without it, the shape of the dense outputs cannot be computed).
            Note that if the recurrent layer is not the first layer
            in your model, you would need to specify the input length
            at the level of the first layer
            (e.g. via the `input_shape` argument)
    # Input shape
        3D tensor with shape `(batch_size, timesteps, input_dim)`.
    # Output shape
        - if `return_state`: a list of tensors. The first tensor is
            the output. The remaining tensors are the last states,
            each with shape `(batch_size, units)`.
        - if `return_sequences`: 3D tensor with shape
            `(batch_size, timesteps, units)`.
        - else, 2D tensor with shape `(batch_size, units)`.
    # Masking
        This layer supports masking for input data with a variable number
        of timesteps. To introduce masks to your data,
        use an [Embedding](embeddings.md) layer with the `mask_zero` parameter
        set to `True`.
    # Note on using statefulness in RNNs
        You can set RNN layers to be 'stateful', which means that the states
        computed for the samples in one batch will be reused as initial states
        for the samples in the next batch. This assumes a one-to-one mapping
        between samples in different successive batches.
        To enable statefulness:
            - specify `stateful=True` in the layer constructor.
            - specify a fixed batch size for your model, by passing
                if sequential model:
                  `batch_input_shape=(...)` to the first layer in your model.
                else for functional model with 1 or more Input layers:
                  `batch_shape=(...)` to all the first layers in your model.
                This is the expected shape of your inputs
                *including the batch size*.
                It should be a tuple of integers, e.g. `(32, 10, 100)`.
            - specify `shuffle=False` when calling fit().
        To reset the states of your model, call `.reset_states()` on either
        a specific layer, or on your entire model.
    # Note on specifying the initial state of RNNs
        You can specify the initial state of RNN layers symbolically by
        calling them with the keyword argument `initial_state`. The value of
        `initial_state` should be a tensor or list of tensors representing
        the initial state of the RNN layer.
        You can specify the initial state of RNN layers numerically by
        calling `reset_states` with the keyword argument `states`. The value of
        `states` should be a numpy array or list of numpy arrays representing
        the initial state of the RNN layer.
    # Note on passing external constants to RNNs
        You can pass "external" constants to the cell using the `constants`
        keyword argument of `RNN.__call__` (as well as `RNN.call`) method. This
        requires that the `cell.call` method accepts the same keyword argument
        `constants`. Such constants can be used to condition the cell
        transformation on additional static inputs (not changing over time),
        a.k.a. an attention mechanism.
    # Examples
    ```python
        # First, let's define a RNN Cell, as a layer subclass.
        class MinimalRNNCell(keras.layers.Layer):
            def __init__(self, units, **kwargs):
                self.units = units
                self.state_size = units
                super(MinimalRNNCell, self).__init__(**kwargs)
            def build(self, input_shape):
                self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                              initializer='uniform',
                                              name='kernel')
                self.recurrent_kernel = self.add_weight(
                    shape=(self.units, self.units),
                    initializer='uniform',
                    name='recurrent_kernel')
                self.built = True
            def call(self, inputs, states):
                prev_output = states[0]
                h = K.dot(inputs, self.kernel)
                output = h + K.dot(prev_output, self.recurrent_kernel)
                return output, [output]
        # Let's use this cell in a RNN layer:
        cell = MinimalRNNCell(32)
        x = keras.Input((None, 5))
        layer = RNN(cell)
        y = layer(x)
        # Here's how to use the cell to build a stacked RNN:
        cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
        x = keras.Input((None, 5))
        layer = RNN(cells)
        y = layer(x)
    ```
    """

    def __init__(self, cell,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if isinstance(cell, (list, tuple)):
            cell = StackedRNNCells(cell)
        if not hasattr(cell, 'call'):
            raise ValueError('`cell` should have a `call` method. '
                             'The RNN was passed:', cell)
        if not hasattr(cell, 'state_size'):
            raise ValueError('The RNN cell should have '
                             'an attribute `state_size` '
                             '(tuple of integers, '
                             'one integer per RNN state).')
        super(RNN, self).__init__(**kwargs)
        self.cell = cell
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll

        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]
        self.state_spec = None
        self._states = None
        self.constants_spec = None
        self._num_constants = None

    @property
    def states(self):
        if self._states is None:
            if isinstance(self.cell.state_size, int):
                num_states = 1
            else:
                num_states = len(self.cell.state_size)
            return [None for _ in range(num_states)]
        return self._states

    @states.setter
    def states(self, states):
        self._states = states

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        if hasattr(self.cell.state_size, '__len__'):
            state_size = self.cell.state_size
        else:
            state_size = [self.cell.state_size]
        output_dim = state_size[0]

        if self.return_sequences:
            output_shape = (input_shape[0], input_shape[1], output_dim)
        else:
            output_shape = (input_shape[0], output_dim)

        if self.return_state:
            state_shape = [(input_shape[0], dim) for dim in state_size]
            return [output_shape] + state_shape
        else:
            return output_shape

    def compute_mask(self, inputs, mask):
        if isinstance(mask, list):
            mask = mask[0]
        output_mask = mask if self.return_sequences else None
        if self.return_state:
            state_mask = [None for _ in self.states]
            return [output_mask] + state_mask
        else:
            return output_mask

    def build(self, input_shape):
        # Note input_shape will be list of shapes of initial states and
        # constants if these are passed in __call__.
        if self._num_constants is not None:
            constants_shape = input_shape[-self._num_constants:]
        else:
            constants_shape = None

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        input_dim = input_shape[-1]
        self.input_spec[0] = InputSpec(shape=(batch_size, None, input_dim))

        # allow cell (if layer) to build before we set or validate state_spec
        if isinstance(self.cell, Layer):
            step_input_shape = (input_shape[0],) + input_shape[2:]
            if constants_shape is not None:
                self.cell.build([step_input_shape] + constants_shape)
            else:
                self.cell.build(step_input_shape)

        # set or validate state_spec
        if hasattr(self.cell.state_size, '__len__'):
            state_size = list(self.cell.state_size)
        else:
            state_size = [self.cell.state_size]

        if self.state_spec is not None:
            # initial_state was passed in call, check compatibility
            if [spec.shape[-1] for spec in self.state_spec] != state_size:
                raise ValueError(
                    'An `initial_state` was passed that is not compatible with '
                    '`cell.state_size`. Received `state_spec`={}; '
                    'however `cell.state_size` is '
                    '{}'.format(self.state_spec, self.cell.state_size))
        else:
            self.state_spec = [InputSpec(shape=(None, dim))
                               for dim in state_size]
        if self.stateful:
            self.reset_states()
        self.built = True

    def get_initial_state(self, inputs):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        if hasattr(self.cell.state_size, '__len__'):
            return [K.tile(initial_state, [1, dim])
                    for dim in self.cell.state_size]
        else:
            return [K.tile(initial_state, [1, self.cell.state_size])]

    def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
        inputs, initial_state, constants = _standardize_args(
            inputs, initial_state, constants, self._num_constants)

        if initial_state is None and constants is None:
            return super(RNN, self).__call__(inputs, **kwargs)

        # If any of `initial_state` or `constants` are specified and are Keras
        # tensors, then add them to the inputs and temporarily modify the
        # input_spec to include them.

        additional_inputs = []
        additional_specs = []
        if initial_state is not None:
            kwargs['initial_state'] = initial_state
            additional_inputs += initial_state
            self.state_spec = [InputSpec(shape=K.int_shape(state))
                               for state in initial_state]
            additional_specs += self.state_spec
        if constants is not None:
            kwargs['constants'] = constants
            additional_inputs += constants
            self.constants_spec = [InputSpec(shape=K.int_shape(constant))
                                   for constant in constants]
            self._num_constants = len(constants)
            additional_specs += self.constants_spec
        # at this point additional_inputs cannot be empty
        is_keras_tensor = K.is_keras_tensor(additional_inputs[0])
        for tensor in additional_inputs:
            if K.is_keras_tensor(tensor) != is_keras_tensor:
                raise ValueError('The initial state or constants of an RNN'
                                 ' layer cannot be specified with a mix of'
                                 ' Keras tensors and non-Keras tensors'
                                 ' (a "Keras tensor" is a tensor that was'
                                 ' returned by a Keras layer, or by `Input`)')

        if is_keras_tensor:
            # Compute the full input spec, including state and constants
            full_input = [inputs] + additional_inputs
            full_input_spec = self.input_spec + additional_specs
            # Perform the call with temporarily replaced input_spec
            original_input_spec = self.input_spec
            self.input_spec = full_input_spec
            output = super(RNN, self).__call__(full_input, **kwargs)
            self.input_spec = original_input_spec
            return output
        else:
            return super(RNN, self).__call__(inputs, **kwargs)

    def call(self,
             inputs,
             mask=None,
             training=None,
             initial_state=None,
             constants=None):
        # input shape: `(samples, time (padded with zeros), input_dim)`
        # note that the .build() method of subclasses MUST define
        # self.input_spec and self.state_spec with complete input shapes.
        if isinstance(inputs, list):
            inputs = inputs[0]
        if initial_state is not None:
            pass
        elif self.stateful:
            initial_state = self.states
        else:
            initial_state = self.get_initial_state(inputs)

        if isinstance(mask, list):
            mask = mask[0]

        if len(initial_state) != len(self.states):
            raise ValueError('Layer has ' + str(len(self.states)) +
                             ' states but was passed ' +
                             str(len(initial_state)) +
                             ' initial states.')
        input_shape = K.int_shape(inputs)
        timesteps = input_shape[1]
        if self.unroll and timesteps in [None, 1]:
            raise ValueError('Cannot unroll a RNN if the '
                             'time dimension is undefined or equal to 1. \n'
                             '- If using a Sequential model, '
                             'specify the time dimension by passing '
                             'an `input_shape` or `batch_input_shape` '
                             'argument to your first layer. If your '
                             'first layer is an Embedding, you can '
                             'also use the `input_length` argument.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a `shape` '
                             'or `batch_shape` argument to your Input layer.')

        kwargs = {}
        if has_arg(self.cell.call, 'training'):
            kwargs['training'] = training

        if constants:
            if not has_arg(self.cell.call, 'constants'):
                raise ValueError('RNN cell does not support constants')

            def step(inputs, states):
                constants = states[-self._num_constants:]
                states = states[:-self._num_constants]
                return self.cell.call(inputs, states, constants=constants,
                                      **kwargs)
        else:
            def step(inputs, states):
                return self.cell.call(inputs, states, **kwargs)

        last_output, outputs, states = K.rnn(step,
                                             inputs,
                                             initial_state,
                                             constants=constants,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             unroll=self.unroll,
                                             input_length=timesteps)
        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, inputs)

        if self.return_sequences:
            output = outputs
        else:
            output = last_output

        # Properly set learning phase
        if getattr(last_output, '_uses_learning_phase', False):
            output._uses_learning_phase = True
            for state in states:
                state._uses_learning_phase = True

        if self.return_state:
            if not isinstance(states, (list, tuple)):
                states = [states]
            else:
                states = list(states)
            return [output] + states
        else:
            return output

    def reset_states(self, states=None):
        if not self.stateful:
            raise AttributeError('Layer must be stateful.')
        batch_size = self.input_spec[0].shape[0]
        if not batch_size:
            raise ValueError('If a RNN is stateful, it needs to know '
                             'its batch size. Specify the batch size '
                             'of your input tensors: \n'
                             '- If using a Sequential model, '
                             'specify the batch size by passing '
                             'a `batch_input_shape` '
                             'argument to your first layer.\n'
                             '- If using the functional API, specify '
                             'the batch size by passing a '
                             '`batch_shape` argument to your Input layer.')
        # initialize state if None
        if self.states[0] is None:
            if hasattr(self.cell.state_size, '__len__'):
                self.states = [K.zeros((batch_size, dim))
                               for dim in self.cell.state_size]
            else:
                self.states = [K.zeros((batch_size, self.cell.state_size))]
        elif states is None:
            if hasattr(self.cell.state_size, '__len__'):
                for state, dim in zip(self.states, self.cell.state_size):
                    K.set_value(state, np.zeros((batch_size, dim)))
            else:
                K.set_value(self.states[0],
                            np.zeros((batch_size, self.cell.state_size)))
        else:
            if not isinstance(states, (list, tuple)):
                states = [states]
            if len(states) != len(self.states):
                raise ValueError('Layer ' + self.name + ' expects ' +
                                 str(len(self.states)) + ' states, '
                                 'but it received ' + str(len(states)) +
                                 ' state values. Input received: ' +
                                 str(states))
            for index, (value, state) in enumerate(zip(states, self.states)):
                if hasattr(self.cell.state_size, '__len__'):
                    dim = self.cell.state_size[index]
                else:
                    dim = self.cell.state_size
                if value.shape != (batch_size, dim):
                    raise ValueError('State ' + str(index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected shape=' +
                                     str((batch_size, dim)) +
                                     ', found shape=' + str(value.shape))
                # TODO: consider batch calls to `set_value`.
                K.set_value(state, value)

    def get_config(self):
        config = {'return_sequences': self.return_sequences,
                  'return_state': self.return_state,
                  'go_backwards': self.go_backwards,
                  'stateful': self.stateful,
                  'unroll': self.unroll}
        if self._num_constants is not None:
            config['num_constants'] = self._num_constants

        cell_config = self.cell.get_config()
        config['cell'] = {'class_name': self.cell.__class__.__name__,
                          'config': cell_config}
        base_config = super(RNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from . import deserialize as deserialize_layer
        cell = deserialize_layer(config.pop('cell'),
                                 custom_objects=custom_objects)
        num_constants = config.pop('num_constants', None)
        layer = cls(cell, **config)
        layer._num_constants = num_constants
        return layer

    @property
    def trainable_weights(self):
        if not self.trainable:
            return []
        if isinstance(self.cell, Layer):
            return self.cell.trainable_weights
        return []

    @property
    def non_trainable_weights(self):
        if isinstance(self.cell, Layer):
            if not self.trainable:
                return self.cell.weights
            return self.cell.non_trainable_weights
        return []

    @property
    def losses(self):
        layer_losses = super(RNN, self).losses
        if isinstance(self.cell, Layer):
            return self.cell.losses + layer_losses
        return layer_losses

    def get_losses_for(self, inputs=None):
        if isinstance(self.cell, Layer):
            cell_losses = self.cell.get_losses_for(inputs)
            return cell_losses + super(RNN, self).get_losses_for(inputs)
        return super(RNN, self).get_losses_for(inputs)


def has_arg(fn, name, accept_all=False):
    """Checks if a callable accepts a given keyword argument.
    For Python 2, checks if there is an argument with the given name.
    For Python 3, checks if there is an argument with the given name, and
    also whether this argument can be called with a keyword (i.e. if it is
    not a positional-only argument).
    # Arguments
        fn: Callable to inspect.
        name: Check if `fn` can be called with `name` as a keyword argument.
        accept_all: What to return if there is no parameter called `name`
                    but the function accepts a `**kwargs` argument.
    # Returns
        bool, whether `fn` accepts a `name` keyword argument.
    """
    if sys.version_info < (3,):
        arg_spec = inspect.getargspec(fn)
        if accept_all and arg_spec.keywords is not None:
            return True
        return (name in arg_spec.args)
    elif sys.version_info < (3, 3):
        arg_spec = inspect.getfullargspec(fn)
        if accept_all and arg_spec.varkw is not None:
            return True
        return (name in arg_spec.args or
                name in arg_spec.kwonlyargs)
    else:
        signature = inspect.signature(fn)
        parameter = signature.parameters.get(name)
        if parameter is None:
            if accept_all:
                for param in signature.parameters.values():
                    if param.kind == inspect.Parameter.VAR_KEYWORD:
                        return True
            return False
        return (parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                   inspect.Parameter.KEYWORD_ONLY))


def _standardize_args(inputs, initial_state, constants, num_constants):
    """Standardize `__call__` to a single list of tensor inputs.
    When running a model loaded from file, the input tensors
    `initial_state` and `constants` can be passed to `RNN.__call__` as part
    of `inputs` instead of by the dedicated keyword arguments. This method
    makes sure the arguments are separated and that `initial_state` and
    `constants` are lists of tensors (or None).
    # Arguments
        inputs: tensor or list/tuple of tensors
        initial_state: tensor or list of tensors or None
        constants: tensor or list of tensors or None
    # Returns
        inputs: tensor
        initial_state: list of tensors or None
        constants: list of tensors or None
    """
    if isinstance(inputs, list):
        assert initial_state is None and constants is None
        if num_constants is not None:
            constants = inputs[-num_constants:]
            inputs = inputs[:-num_constants]
        if len(inputs) > 1:
            initial_state = inputs[1:]
        inputs = inputs[0]

    def to_list_or_none(x):
        if x is None or isinstance(x, list):
            return x
        if isinstance(x, tuple):
            return list(x)
        return [x]

    initial_state = to_list_or_none(initial_state)
    constants = to_list_or_none(constants)

    return inputs, initial_state, constants
