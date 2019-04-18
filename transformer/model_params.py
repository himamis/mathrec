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
"""Defines Transformer model parameters."""

from collections import defaultdict


BASE_PARAMS = defaultdict(
    lambda: None,  # Set default value to None.

    add_position_timing_signal=False,
    add_step_timing_signal=False,
    step_timing_signal_type="sinusoid",  # "learned",
    position_start_index=None,
    add_or_concat_timing_signal="add",

    # Input params
    default_batch_size=2048,  # Maximum number of tokens per batch of examples.
    default_batch_size_tpu=32768,
    max_length=256,  # Maximum number of tokens per example.

    # Model params
    initializer_gain=1.0,  # Used in trainable variable initialization.
    vocab_size=33708,  # Number of tokens defined in the vocabulary file.
    hidden_size=512,  # Model dimension in the hidden layers.
    num_hidden_layers=6,  # Number of layers in the encoder and decoder stacks.
    num_heads=8,  # Number of heads to use in multi-headed attention.
    filter_size=2048,  # Inner layer dimension in the feedforward network.

    # Dropout values (only used when training)
    layer_postprocess_dropout=0.1,
    attention_dropout=0.1,
    relu_dropout=0.1,

    # Training params
    label_smoothing=0.1,
    learning_rate=2.0,
    learning_rate_decay_rate=1.0,
    learning_rate_warmup_steps=16000,

    # Optimizer params
    optimizer_adam_beta1=0.9,
    optimizer_adam_beta2=0.997,
    optimizer_adam_epsilon=1e-09,

    # Default prediction params
    extra_decode_length=50,
    beam_size=4,
    alpha=0.6,  # used to calculate length normalization in beam search

    # TPU specific parameters
    use_tpu=False,
    static_batch=False,
    allow_ffn_pad=True,

    # Beta
    beta=0
)

BIG_PARAMS = BASE_PARAMS.copy()
BIG_PARAMS.update(
    default_batch_size=4096,

    # default batch size is smaller than for BASE_PARAMS due to memory limits.
    default_batch_size_tpu=16384,

    hidden_size=1024,
    filter_size=4096,
    num_heads=16,
)

# Parameters for running the model in multi gpu. These should not change the
# params that modify the model shape (such as the hidden_size or num_heads).
BASE_MULTI_GPU_PARAMS = BASE_PARAMS.copy()
BASE_MULTI_GPU_PARAMS.update(
    learning_rate_warmup_steps=8000
)

BIG_MULTI_GPU_PARAMS = BIG_PARAMS.copy()
BIG_MULTI_GPU_PARAMS.update(
    layer_postprocess_dropout=0.3,
    learning_rate_warmup_steps=8000
)

# Parameters for testing the model
TINY_PARAMS = BASE_PARAMS.copy()
TINY_PARAMS.update(
    default_batch_size=1024,
    default_batch_size_tpu=1024,
    hidden_size=32,
    num_heads=4,
    filter_size=256,
)

CUSTOM_PARAMS = BASE_PARAMS.copy()
CUSTOM_PARAMS.update(
    num_hidden_layers=6,
    hidden_size=512,
    num_heads=8,
    learning_rate_warmup_steps=14000,
    learning_rate=2.0,
    layer_postprocess_dropout=0.2,
    attention_dropout=0.2,
    relu_dropout=0.2,

    beam_size=6,
    alpha=0.6,
    reduce_distance=None,

    transform_diffs=True,
    l2_regularization=0.0001,

    act_type="basic",
    act_epsilon=0.01,
    act_max_steps=12,
    act_halting_bias_init=1.0,
    act_loss_weight=0.01,
    num_inrecurrence_layers=1,
#    layer_preprocess_sequence=
##    layer_prepostprocess_dropout=
 #   norm_type=
 #   norm_epsilon=
 #   layer_prepostprocess_dropout_broadcast_dims=
    # learning_rate=0.0007
)
