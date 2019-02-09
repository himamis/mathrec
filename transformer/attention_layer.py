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
"""Implementation of multiheaded attention and self-attention layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Attention(tf.layers.Layer):
    """Multi-headed attention layer."""

    def __init__(self, hidden_size, num_heads, attention_dropout, train, diffs_att=False):
        if hidden_size % num_heads != 0:
            raise ValueError("Hidden size must be evenly divisible by the number of "
                             "heads.")

        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = self.hidden_size // self.num_heads
        self.attention_dropout = attention_dropout
        self.train = train
        self.diffs_att = diffs_att

        # Layers for linearly projecting the queries, keys, and values.
        self.q_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="q")
        self.k_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="k")
        self.v_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="v")
        if diffs_att:
            self.bk_dense_layer = tf.layers.Dense(hidden_size, use_bias=True, name="bk")
            self.bv_dense_layer = tf.layers.Dense(hidden_size, use_bias=True, name="bv")

        self.output_dense_layer = tf.layers.Dense(hidden_size, use_bias=False,
                                                  name="output_transform")

    def split_heads(self, x):
        """Split x into different heads, and transpose the resulting value.

    The tensor is transposed to insure the inner dimensions hold the correct
    values during the matrix multiplication.

    Args:
      x: A tensor with shape [batch_size, length, hidden_size]

    Returns:
      A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
    """
        with tf.name_scope("split_heads"):
            x_shape = tf.shape(x)
            batch_size = x_shape[0]
            length = x_shape[1]

            # Split the last dimension
            x = tf.reshape(x, [batch_size, length, self.num_heads, self.depth])

            # Transpose the result
            return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        """Combine tensor that has been split.

    Args:
      x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]

    Returns:
      A tensor with shape [batch_size, length, hidden_size]
    """
        with tf.name_scope("combine_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[2]
            x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
            return tf.reshape(x, [batch_size, length, self.hidden_size])

    def call(self, x, y, bias, cache=None, **kwargs):
        """Apply attention mechanism to x and y.

        Args:
      x: a tensor with shape [batch_size, length_x, hidden_size]
      y: a tensor with shape [batch_size, length_y, hidden_size]
      bias: attention bias that will be added to the result of the dot product.
      cache: (Used during prediction) dictionary with tensors containing results
        of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, key_channels],
             "v": tensor with shape [batch_size, i, value_channels]}
        where i is the current decoded length.

    Returns:
      Attention layer output with shape [batch_size, length_x, hidden_size]
    """


        # Linearly project the query (q), key (k) and value (v) using different
        # learned projections. This is in preparation of splitting them into
        # multiple heads. Multi-head attention uses multiple queries, keys, and
        # values rather than regular attention (which uses a single q, k, v).
        q = self.q_dense_layer(x)
        k = self.k_dense_layer(y)
        v = self.v_dense_layer(y)
        if self.diffs_att:
            assert "diffs" in kwargs, "Must provide `diffs` argument when using diffs_att"
            b = kwargs["diffs"]  # [batch_size, length, length, 4]
            bk = self.bk_dense_layer(b)
            bv = self.bv_dense_layer(b)

        if cache is not None:
            # Combine cached keys and values with new keys and values.
            k = tf.concat([cache["k"], k], axis=1)
            v = tf.concat([cache["v"], v], axis=1)

            # Update cache
            cache["k"] = k
            cache["v"] = v

        # Split q, k, v into heads.
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        if self.diffs_att:
            bk = self._split_reshape_b(bk)
            bv = self._split_reshape_b(bv)

        # Scale q to prevent the dot product between q and k from growing too large.
        q *= self.depth ** -0.5

        # Calculate dot product attention
        if self.diffs_att:
            logits = self._relative_attention_inner(q, k, bk, True)
        else:
            logits = tf.matmul(q, k, transpose_b=True)

        logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights")
        if self.train:
            weights = tf.nn.dropout(weights, 1.0 - self.attention_dropout)

        if self.diffs_att:
            attention_output = self._relative_attention_inner(weights, v, bv, False)
        else:
            attention_output = tf.matmul(weights, v)

        # Recombine heads --> [batch_size, length, hidden_size]
        attention_output = self.combine_heads(attention_output)

        # Run the combined outputs through another linear projection layer.
        attention_output = self.output_dense_layer(attention_output)
        return attention_output


    def _relative_attention_inner(self, x, y, z, transpose):
        """Relative position-aware dot-product attention inner calculation.

        This batches matrix multiply calculations to avoid unnecessary broadcasting.

        Args:
          x: Tensor with shape [batch_size, heads, length or 1, length or depth].
          y: Tensor with shape [batch_size, heads, length or 1, depth].
          z: Tensor with shape [length or 1, length, depth].
          transpose: Whether to transpose inner matrices of y and z. Should be true if
              last dimension of x is depth, not length.

        Returns:
          A Tensor with shape [batch_size, heads, length, length or depth].
        """
        batch_size = tf.shape(x)[0]
        heads = x.get_shape().as_list()[1]
        length = tf.shape(x)[2]

        # xy_matmul is [batch_size, heads, length or 1, length or depth]
        xy_matmul = tf.matmul(x, y, transpose_b=transpose)
        # x_t is [length or 1, batch_size, heads, length or depth]
        x_t = tf.transpose(x, [2, 0, 1, 3])
        # x_t_r is [length or 1, batch_size * heads, 1, length or depth]
        x_t_r = tf.reshape(x_t, [length, heads * batch_size, 1, -1])
        # x_tz_matmul is [length or 1, batch_size * heads, length or depth]
        x_tz_matmul = tf.matmul(x_t_r, z, transpose_b=transpose)
        # x_tz_matmul_r is [length or 1, batch_size, heads, length or depth]
        x_tz_matmul_r = tf.reshape(x_tz_matmul, [length, batch_size, heads, -1])
        # x_tz_matmul_r_t is [batch_size, heads, length or 1, length or depth]
        x_tz_matmul_r_t = tf.transpose(x_tz_matmul_r, [1, 2, 0, 3])
        return xy_matmul + x_tz_matmul_r_t

    def _split_reshape_b(self, b):
        with tf.name_scope("split_heads"):
            b_shape = tf.shape(b)
            batch_size = b_shape[0]
            length = b_shape[1]

            # Split the last dimension
            b = tf.reshape(b, [batch_size, length, length, self.num_heads, self.depth])

            # Transpose and reshape the result
            b = tf.transpose(b, [1, 0, 3, 2, 4])
            b = tf.reshape(b, [length, self.num_heads * batch_size, length, -1])

            return b


class SelfAttention(Attention):
    """Multiheaded self-attention layer."""

    def call(self, x, bias, cache=None, **kwargs):
        return super(SelfAttention, self).call(x, x, bias, cache, **kwargs)
