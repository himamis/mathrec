# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Transformer model helper methods."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

_NEG_INF = -1e9


def get_position_encoding(
        length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    """Return positional encoding.

    Calculates the position encoding as a mix of sine and cosine functions with
    geometrically increasing wavelengths.
    Defined and formulized in Attention is All You Need, section 3.5.

    Args:
      length: Sequence length.
      hidden_size: Size of the
      min_timescale: Minimum scale that will be applied at each position
      max_timescale: Maximum scale that will be applied at each position

    Returns:
      Tensor with shape [length, hidden_size]
    """
    position = tf.to_float(tf.range(length))
    num_timescales = hidden_size // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal


def get_decoder_self_attention_bias(length):
    """Calculate bias for decoder that maintains model's autoregressive property.

    Creates a tensor that masks out locations that correspond to illegal
    connections, so prediction at position i cannot draw information from future
    positions.

    Args:
      length: int length of sequences in batch.

    Returns:
      float tensor of shape [1, 1, length, length]
    """
    with tf.name_scope("decoder_self_attention_bias"):
        valid_locs = tf.matrix_band_part(tf.ones([length, length]), -1, 0)
        valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
        decoder_bias = _NEG_INF * (1.0 - valid_locs)
    return decoder_bias


def get_padding(x, padding_value=0):
    """Return float tensor representing the padding values in x.

    Args:
      x: int tensor with any shape
      padding_value: int value that

    Returns:
      flaot tensor with same shape as x containing values 0 or 1.
        0 -> non-padding, 1 -> padding
    """
    with tf.name_scope("padding"):
        return tf.to_float(tf.equal(x, padding_value))


def get_padding_bias(x):
    """Calculate bias tensor from padding values in tensor.

    Bias tensor that is added to the pre-softmax multi-headed attention logits,
    which has shape [batch_size, num_heads, length, length]. The tensor is zero at
    non-padding locations, and -1e9 (negative infinity) at padding locations.

    Args:
      x: int tensor with shape [batch_size, length]

    Returns:
      Attention bias tensor of shape [batch_size, 1, 1, length].
    """
    with tf.name_scope("attention_bias"):
        padding = get_padding(x)
        attention_bias = padding * _NEG_INF
        attention_bias = tf.expand_dims(
            tf.expand_dims(attention_bias, axis=1), axis=1)
    return attention_bias


def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i in range(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


def get_timing_signal_1d(length,
                         channels,
                         min_timescale=1.0,
                         max_timescale=1.0e4,
                         start_index=0):
    """Gets a bunch of sinusoids of different frequencies.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    expressed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
      length: scalar, length of timing signal sequence.
      channels: scalar, size of timing embeddings to create. The number of
          different timescales is equal to channels / 2.
      min_timescale: a float
      max_timescale: a float
      start_index: index of first position
    Returns:
      a Tensor of timing signals [1, length, channels]
    """
    position = tf.to_float(tf.range(length) + start_index)
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            tf.maximum(tf.to_float(num_timescales) - 1, 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal


def get_layer_timing_signal_sinusoid_1d(channels, layer, num_layers):
    """Add sinusoids of different frequencies as layer (vertical) timing signal.
    Args:
      channels: dimension of the timing signal
      layer: layer num
      num_layers: total number of layers
    Returns:
      a Tensor of timing signals [1, 1, channels].
    """

    signal = get_timing_signal_1d(num_layers, channels)
    layer_signal = tf.expand_dims(signal[:, layer, :], axis=1)

    return layer_signal


def get_layer_timing_signal_learned_1d(channels, layer, num_layers):
    """get n-dimensional embedding as the layer (vertical) timing signal.
    Adds embeddings to represent the position of the layer in the tower.
    Args:
      channels: dimension of the timing signal
      layer: layer num
      num_layers: total number of layers
    Returns:
      a Tensor of timing signals [1, 1, channels].
    """
    shape = [num_layers, 1, 1, channels]
    layer_embedding = (
            tf.get_variable(
                "layer_embedding",
                shape,
                initializer=tf.random_normal_initializer(0, channels ** -0.5)) *
            (channels ** 0.5))
    return layer_embedding[layer, :, :, :]


def cast_like(x, y):
    """Cast x to y's dtype, if necessary."""
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)

    if x.dtype.base_dtype == y.dtype.base_dtype:
        return x

    cast_x = tf.cast(x, y.dtype)
    if cast_x.device != x.device:
        tf.logging.warning("Cast for %s may induce copy from '%s' to '%s'", x.name,
                           x.device, cast_x.device)
    return cast_x


def add_position_timing_signal(x, step, hparams):
    """Add n-dimensional embedding as the position (horizontal) timing signal.
    Args:
      x: a tensor with shape [batch, length, depth]
      step: step
      hparams: model hyper parameters
    Returns:
      a Tensor with the same shape as x.
    """

    if not hparams["position_start_index"]:
        index = 0

    elif hparams["position_start_index"] == "random":
        # Shift all positions randomly
        # TODO(dehghani): What would be reasonable for max number of shift?
        index = tf.random_uniform(
            [], maxval=shape_list(x)[1], dtype=tf.int32)

    elif hparams["position_start_index"] == "step":
        # Shift positions based on the step
        if hparams["recurrence_type"] == "act":
            num_steps = hparams["act_max_steps"]
        else:
            num_steps = hparams["num_rec_steps"]
        index = tf.cast(
            shape_list(x)[1] * step / num_steps, dtype=tf.int32)

    # No need for the timing signal in the encoder/decoder input preparation
    assert hparams["pos"] is None

    length = shape_list(x)[1]
    channels = shape_list(x)[2]
    signal = get_timing_signal_1d(
        length, channels, start_index=index)

    if hparams["add_or_concat_timing_signal"] == "add":
        x_with_timing = x + cast_like(signal, x)

    elif hparams["add_or_concat_timing_signal"] == "concat":
        batch_size = shape_list(x)[0]
        signal_tiled = tf.tile(signal, [batch_size, 1, 1])
        x_with_timing = tf.concat((x, signal_tiled), axis=-1)

    return x_with_timing


def add_step_timing_signal(x, step, hparams):
    """Add n-dimensional embedding as the step (vertical) timing signal.
    Args:
      x: a tensor with shape [batch, length, depth]
      step: step
      hparams: model hyper parameters
    Returns:
      a Tensor with the same shape as x.
    """
    # if hparams["recurrence_type"] == "act":
    #     num_steps = hparams["act_max_steps"]
    # else:
    #     num_steps = hparams["num_rec_steps"]
    num_steps = hparams["num_hidden_layers"]
    channels = shape_list(x)[-1]

    if hparams["step_timing_signal_type"] == "learned":
        signal = get_layer_timing_signal_learned_1d(
            channels, step, num_steps)

    elif hparams["step_timing_signal_type"] == "sinusoid":
        signal = get_layer_timing_signal_sinusoid_1d(
            channels, step, num_steps)

    if hparams["add_or_concat_timing_signal"] == "add":
        x_with_timing = x + cast_like(signal, x)

    elif hparams["add_or_concat_timing_signal"] == "concat":
        batch_size = shape_list(x)[0]
        length = shape_list(x)[1]
        signal_tiled = tf.tile(signal, [batch_size, length, 1])
        x_with_timing = tf.concat((x, signal_tiled), axis=-1)

    return x_with_timing
