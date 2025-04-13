# Copyright 2024 The Magenta Authors.
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

"""SketchRNN RNN definition."""
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


def orthogonal(shape):
  """Orthogonal initializer."""
  flat_shape = (shape[0], np.prod(shape[1:]))
  a = np.random.normal(0.0, 1.0, flat_shape)
  u, _, v = np.linalg.svd(a, full_matrices=False)
  q = u if u.shape == flat_shape else v
  return q.reshape(shape)


def orthogonal_initializer(scale=1.0):
  """Orthogonal initializer."""
  def _initializer(shape, dtype=tf.float32,
                   partition_info=None):  # pylint: disable=unused-argument
    return tf.constant(orthogonal(shape) * scale, dtype)

  return _initializer


def lstm_ortho_initializer(scale=1.0):
  """LSTM orthogonal initializer."""
  def _initializer(shape, dtype=tf.float32,
                   partition_info=None):  # pylint: disable=unused-argument
    size_x = shape[0]
    size_h = shape[1] // 4  # assumes lstm.
    t = np.zeros(shape)
    t[:, :size_h] = orthogonal([size_x, size_h]) * scale
    t[:, size_h:size_h * 2] = orthogonal([size_x, size_h]) * scale
    t[:, size_h * 2:size_h * 3] = orthogonal([size_x, size_h]) * scale
    t[:, size_h * 3:] = orthogonal([size_x, size_h]) * scale
    return tf.constant(t, dtype)

  return _initializer

def damage_smallest(weights, p_smallest, name='smallest_damaged_weights'):
  shape = weights.get_shape().as_list()
  num_weights = np.prod(shape)
  num_damage = int(p_smallest * num_weights)

  # get indices of smallest magnitude weights
  flat_weights = tf.reshape(tf.abs(weights), [-1])
  _, indices = tf.mat.top_k(-flat_weights, k=num_damage)

  mask = tf.ones_like(flat_weights, dtype=tf.float32)
  update = tf.zeros_like(indices, dtype=tf.float32)
  mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(indices, 1), update) # set smallest weights to 0
  mask = tf.reshape(mask, shape, name=name) # reshape mask to original shape

  # apply mask to weights
  return weights * mask

def damage_fas(weights, p_block, p_reflect, p_filter, name='fas_damaged_weights'):
  if p_block + p_reflect + p_filter > 1:
    raise ValueError("percentages for fas damage types must not exceed 100%")
  
  flat_weights = tf.reshape(tf.abs(weights), [-1])
  nonzero_indices = tf.where(tf.not_equal(tf.reshape(weights, [-1]), 0))
  nonzero_values = tf.gather(flat_weights, nonzero_indices)
  num_nonzero_indices = tf.math.count_nonzero(weights)
  
  # percentage of weights damaged will be taken from the number of nonzero weights
  # simulated fas damage occurs after energy constraint blockage
  num_block = int(num_nonzero_indices * p_block)
  num_reflect = int(num_nonzero_indices * p_reflect)
  num_filter = int(num_nonzero_indices * p_filter)

  shuffled_indices = tf.random.shuffle(nonzero_indices)

  indices_block = shuffled_indices[:num_block]
  indices_reflect = shuffled_indices[num_block:num_block+num_reflect]
  indices_filter = shuffled_indices[num_block+num_reflect:num_block+num_reflect+num_filter]

  # do damage
  damaged_weights = flat_weights

  # blockage: set weights to 0
  damaged_weights = tf.tensor_scatter_nd_update(damaged_weights, indices_block, tf.zeros([num_block]), dtype=weights.dtype)
  # reflect: halve weights
  weights_to_reflect = tf.gather(damaged_weights, indices_reflect)
  damaged_weights = tf.tensor_scatter_nd_update(damaged_weights, indices_reflect, weights_to_reflect * 0.5)
  # filter: low pass filter (lusch et al)
  weights_to_filter = tf.father(damaged_weights, indices_filter)                    # get weights before transformation
  signs = tf.sign(weights_to_filter)                                                # get signs of weights
  high_weight = tfp.stats.percentile(flat_weights, 95.0, interpolation='linear')    # get high_weight, should be in the 95th percentile for all weights
  normalized_weights = tf.divide(weights_to_filter, high_weight)                    # scale weights to mostly between -1 and 1
  x = normalized_weights
  transformed_weights = -0.2744 * tf.pow(x, 2) + 0.9094 * x - 0.0192
  transformed_weights = transformed_weights + tf.random.normal(tf.shape(transformed_weights), mean=0.0, stddev=0.05)
  filtered_weights = transformed_weights * signs * high_weight                      # rescale
  damaged_weights = tf.tensor_scatter_nd_update(damaged_weights, indices_filter, filtered_weights)

  fas_damaged_weights = tf.reshape(damaged_weights, tf.shape(weights), name=name)
  return fas_damaged_weights


class LSTMCell(tf.nn.rnn_cell.RNNCell):
  """Vanilla LSTM cell.

  Uses ortho initializer, and also recurrent dropout without memory loss
  (https://arxiv.org/abs/1603.05118)
  """

  def __init__(self,
               num_units,
               p_smallest, # !!
               forget_bias=1.0,
               use_recurrent_dropout=False,
               dropout_keep_prob=0.9):
    self.num_units = num_units
    self.forget_bias = forget_bias
    self.use_recurrent_dropout = use_recurrent_dropout
    self.dropout_keep_prob = dropout_keep_prob
    self.p_smallest = p_smallest # !!

  @property
  def state_size(self):
    return 2 * self.num_units

  @property
  def output_size(self):
    return self.num_units

  def get_output(self, state):
    unused_c, h = tf.split(state, 2, 1)
    return h

  def __call__(self, x, state, scope=None):
    with tf.variable_scope(scope or type(self).__name__):
      c, h = tf.split(state, 2, 1)

      x_size = x.get_shape().as_list()[1]

      w_init = None  # uniform

      h_init = lstm_ortho_initializer(1.0)

      # Keep W_xh and W_hh separate here as well to use different init methods.
      w_xh = tf.get_variable(
          'W_xh', [x_size, 4 * self.num_units], initializer=w_init)
      w_hh = tf.get_variable(
          'W_hh', [self.num_units, 4 * self.num_units], initializer=h_init)
      bias = tf.get_variable(
          'bias', [4 * self.num_units],
          initializer=tf.constant_initializer(0.0))
      
      # damage weights!!
      # target only connections between neurons and not bias weights
      w_xh = damage_smallest(w_xh, self.p_smallest)
      w_hh = damage_smallest(w_hh, self.p_smallest)

      concat = tf.concat([x, h], 1)
      w_full = tf.concat([w_xh, w_hh], 0)
      hidden = tf.matmul(concat, w_full) + bias

      i, j, f, o = tf.split(hidden, 4, 1)

      if self.use_recurrent_dropout:
        g = tf.nn.dropout(tf.tanh(j), self.dropout_keep_prob)
      else:
        g = tf.tanh(j)

      new_c = c * tf.sigmoid(f + self.forget_bias) + tf.sigmoid(i) * g
      new_h = tf.tanh(new_c) * tf.sigmoid(o)

      return new_h, tf.concat([new_c, new_h], 1)  # fuk tuples.


def layer_norm_all(h,
                   batch_size,
                   base,
                   num_units,
                   scope='layer_norm',
                   reuse=False,
                   gamma_start=1.0,
                   epsilon=1e-3,
                   use_bias=True):
  """Layer Norm (faster version, but not using defun)."""
  # Performs layer norm on multiple base at once (ie, i, g, j, o for lstm)
  # Reshapes h in to perform layer norm in parallel
  h_reshape = tf.reshape(h, [batch_size, base, num_units])
  mean = tf.reduce_mean(h_reshape, [2], keep_dims=True)
  var = tf.reduce_mean(tf.square(h_reshape - mean), [2], keep_dims=True)
  epsilon = tf.constant(epsilon)
  rstd = tf.rsqrt(var + epsilon)
  h_reshape = (h_reshape - mean) * rstd
  # reshape back to original
  h = tf.reshape(h_reshape, [batch_size, base * num_units])
  with tf.variable_scope(scope):
    if reuse:
      tf.get_variable_scope().reuse_variables()
    gamma = tf.get_variable(
        'ln_gamma', [4 * num_units],
        initializer=tf.constant_initializer(gamma_start))
    if use_bias:
      beta = tf.get_variable(
          'ln_beta', [4 * num_units], initializer=tf.constant_initializer(0.0))
  if use_bias:
    return gamma * h + beta
  return gamma * h


def layer_norm(x,
               num_units,
               scope='layer_norm',
               reuse=False,
               gamma_start=1.0,
               epsilon=1e-3,
               use_bias=True):
  """Calculate layer norm."""
  axes = [1]
  mean = tf.reduce_mean(x, axes, keep_dims=True)
  x_shifted = x - mean
  var = tf.reduce_mean(tf.square(x_shifted), axes, keep_dims=True)
  inv_std = tf.rsqrt(var + epsilon)
  with tf.variable_scope(scope):
    if reuse:
      tf.get_variable_scope().reuse_variables()
    gamma = tf.get_variable(
        'ln_gamma', [num_units],
        initializer=tf.constant_initializer(gamma_start))
    if use_bias:
      beta = tf.get_variable(
          'ln_beta', [num_units], initializer=tf.constant_initializer(0.0))
  output = gamma * (x_shifted) * inv_std
  if use_bias:
    output += beta
  return output


def raw_layer_norm(x, epsilon=1e-3):
  axes = [1]
  mean = tf.reduce_mean(x, axes, keep_dims=True)
  std = tf.sqrt(
      tf.reduce_mean(tf.square(x - mean), axes, keep_dims=True) + epsilon)
  output = (x - mean) / (std)
  return output


def super_linear(x,
                 output_size,
                 scope=None,
                 reuse=False,
                 init_w='ortho',
                 weight_start=0.0,
                 use_bias=True,
                 bias_start=0.0,
                 input_size=None):
  """Performs linear operation. Uses ortho init defined earlier."""
  shape = x.get_shape().as_list()
  with tf.variable_scope(scope or 'linear'):
    if reuse:
      tf.get_variable_scope().reuse_variables()

    w_init = None  # uniform
    if input_size is None:
      x_size = shape[1]
    else:
      x_size = input_size
    if init_w == 'zeros':
      w_init = tf.constant_initializer(0.0)
    elif init_w == 'constant':
      w_init = tf.constant_initializer(weight_start)
    elif init_w == 'gaussian':
      w_init = tf.random_normal_initializer(stddev=weight_start)
    elif init_w == 'ortho':
      w_init = lstm_ortho_initializer(1.0)

    w = tf.get_variable(
        'super_linear_w', [x_size, output_size], tf.float32, initializer=w_init)
    if use_bias:
      b = tf.get_variable(
          'super_linear_b', [output_size],
          tf.float32,
          initializer=tf.constant_initializer(bias_start))
      return tf.matmul(x, w) + b
    return tf.matmul(x, w)


class LayerNormLSTMCell(tf.nn.rnn_cell.RNNCell):
  """Layer-Norm, with Ortho Init. and Recurrent Dropout without Memory Loss.

  https://arxiv.org/abs/1607.06450 - Layer Norm
  https://arxiv.org/abs/1603.05118 - Recurrent Dropout without Memory Loss
  """

  def __init__(self,
               num_units,
               p_smallest, # !!
               forget_bias=1.0,
               use_recurrent_dropout=False,
               dropout_keep_prob=0.90):
    """Initialize the Layer Norm LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (default 1.0).
      use_recurrent_dropout: Whether to use Recurrent Dropout (default False)
      dropout_keep_prob: float, dropout keep probability (default 0.90)
    """
    self.num_units = num_units
    self.forget_bias = forget_bias
    self.use_recurrent_dropout = use_recurrent_dropout
    self.dropout_keep_prob = dropout_keep_prob
    self.p_smallest = p_smallest # !!

  @property
  def input_size(self):
    return self.num_units

  @property
  def output_size(self):
    return self.num_units

  @property
  def state_size(self):
    return 2 * self.num_units

  def get_output(self, state):
    h, unused_c = tf.split(state, 2, 1)
    return h

  def __call__(self, x, state, timestep=0, scope=None):
    with tf.variable_scope(scope or type(self).__name__):
      h, c = tf.split(state, 2, 1)

      h_size = self.num_units
      x_size = x.get_shape().as_list()[1]
      batch_size = x.get_shape().as_list()[0]

      w_init = None  # uniform

      h_init = lstm_ortho_initializer(1.0)

      w_xh = tf.get_variable(
          'W_xh', [x_size, 4 * self.num_units], initializer=w_init)
      w_hh = tf.get_variable(
          'W_hh', [self.num_units, 4 * self.num_units], initializer=h_init)
      
      # damage weights!!
      # target only connections between neurons and not bias weights
      w_xh = damage_smallest(w_xh, self.p_smallest)
      w_hh = damage_smallest(w_hh, self.p_smallest)

      concat = tf.concat([x, h], 1)  # concat for speed.
      w_full = tf.concat([w_xh, w_hh], 0)
      concat = tf.matmul(concat, w_full)  #+ bias # live life without garbage.

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      concat = layer_norm_all(concat, batch_size, 4, h_size, 'ln_all')
      i, j, f, o = tf.split(concat, 4, 1)

      if self.use_recurrent_dropout:
        g = tf.nn.dropout(tf.tanh(j), self.dropout_keep_prob)
      else:
        g = tf.tanh(j)

      new_c = c * tf.sigmoid(f + self.forget_bias) + tf.sigmoid(i) * g
      new_h = tf.tanh(layer_norm(new_c, h_size, 'ln_c')) * tf.sigmoid(o)

    return new_h, tf.concat([new_h, new_c], 1)


class HyperLSTMCell(tf.nn.rnn_cell.RNNCell):
  """HyperLSTM with Ortho Init, Layer Norm, Recurrent Dropout, no Memory Loss.

  https://arxiv.org/abs/1609.09106
  http://blog.otoro.net/2016/09/28/hyper-networks/
  """

  def __init__(self,
               num_units,
               p_smallest, # !!
               forget_bias=1.0,
               use_recurrent_dropout=False,
               dropout_keep_prob=0.90,
               use_layer_norm=True,
               hyper_num_units=256,
               hyper_embedding_size=32,
               hyper_use_recurrent_dropout=False):
    """Initialize the Layer Norm HyperLSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (default 1.0).
      use_recurrent_dropout: Whether to use Recurrent Dropout (default False)
      dropout_keep_prob: float, dropout keep probability (default 0.90)
      use_layer_norm: boolean. (default True)
        Controls whether we use LayerNorm layers in main LSTM & HyperLSTM cell.
      hyper_num_units: int, number of units in HyperLSTM cell.
        (default is 128, recommend experimenting with 256 for larger tasks)
      hyper_embedding_size: int, size of signals emitted from HyperLSTM cell.
        (default is 16, recommend trying larger values for large datasets)
      hyper_use_recurrent_dropout: boolean. (default False)
        Controls whether HyperLSTM cell also uses recurrent dropout.
        Recommend turning this on only if hyper_num_units becomes large (>= 512)
    """
    self.num_units = num_units
    self.forget_bias = forget_bias
    self.use_recurrent_dropout = use_recurrent_dropout
    self.dropout_keep_prob = dropout_keep_prob
    self.use_layer_norm = use_layer_norm
    self.hyper_num_units = hyper_num_units
    self.hyper_embedding_size = hyper_embedding_size
    self.hyper_use_recurrent_dropout = hyper_use_recurrent_dropout

    self.total_num_units = self.num_units + self.hyper_num_units
    self.p_smallest = p_smallest # !!

    if self.use_layer_norm:
      cell_fn = LayerNormLSTMCell
    else:
      cell_fn = LSTMCell
    self.hyper_cell = cell_fn(
        hyper_num_units,
        p_smallest, # !!
        use_recurrent_dropout=hyper_use_recurrent_dropout,
        dropout_keep_prob=dropout_keep_prob)

  @property
  def input_size(self):
    return self._input_size

  @property
  def output_size(self):
    return self.num_units

  @property
  def state_size(self):
    return 2 * self.total_num_units

  def get_output(self, state):
    total_h, unused_total_c = tf.split(state, 2, 1)
    h = total_h[:, 0:self.num_units]
    return h

  def hyper_norm(self, layer, scope='hyper', use_bias=True):
    num_units = self.num_units
    embedding_size = self.hyper_embedding_size
    # recurrent batch norm init trick (https://arxiv.org/abs/1603.09025).
    init_gamma = 0.10  # cooijmans' da man.
    with tf.variable_scope(scope):
      zw = super_linear(
          self.hyper_output,
          embedding_size,
          init_w='constant',
          weight_start=0.00,
          use_bias=True,
          bias_start=1.0,
          scope='zw')
      alpha = super_linear(
          zw,
          num_units,
          init_w='constant',
          weight_start=init_gamma / embedding_size,
          use_bias=False,
          scope='alpha')
      result = tf.multiply(alpha, layer)
      if use_bias:
        zb = super_linear(
            self.hyper_output,
            embedding_size,
            init_w='gaussian',
            weight_start=0.01,
            use_bias=False,
            bias_start=0.0,
            scope='zb')
        beta = super_linear(
            zb,
            num_units,
            init_w='constant',
            weight_start=0.00,
            use_bias=False,
            scope='beta')
        result += beta
    return result

  def __call__(self, x, state, timestep=0, scope=None):
    with tf.variable_scope(scope or type(self).__name__):
      total_h, total_c = tf.split(state, 2, 1)
      h = total_h[:, 0:self.num_units]
      c = total_c[:, 0:self.num_units]
      self.hyper_state = tf.concat(
          [total_h[:, self.num_units:], total_c[:, self.num_units:]], 1)

      batch_size = x.get_shape().as_list()[0]
      x_size = x.get_shape().as_list()[1]
      self._input_size = x_size

      w_init = None  # uniform

      h_init = lstm_ortho_initializer(1.0)

      w_xh = tf.get_variable(
          'W_xh', [x_size, 4 * self.num_units], initializer=w_init)
      w_hh = tf.get_variable(
          'W_hh', [self.num_units, 4 * self.num_units], initializer=h_init)
      bias = tf.get_variable(
          'bias', [4 * self.num_units],
          initializer=tf.constant_initializer(0.0))
      
      # damage weights!!
      # target only connections between neurons and not bias weights
      w_xh = damage_smallest(w_xh, self.p_smallest)
      w_hh = damage_smallest(w_hh, self.p_smallest)

      # concatenate the input and hidden states for hyperlstm input
      hyper_input = tf.concat([x, h], 1)
      hyper_output, hyper_new_state = self.hyper_cell(hyper_input,
                                                      self.hyper_state)
      self.hyper_output = hyper_output
      self.hyper_state = hyper_new_state

      xh = tf.matmul(x, w_xh)
      hh = tf.matmul(h, w_hh)

      # split Wxh contributions
      ix, jx, fx, ox = tf.split(xh, 4, 1)
      ix = self.hyper_norm(ix, 'hyper_ix', use_bias=False)
      jx = self.hyper_norm(jx, 'hyper_jx', use_bias=False)
      fx = self.hyper_norm(fx, 'hyper_fx', use_bias=False)
      ox = self.hyper_norm(ox, 'hyper_ox', use_bias=False)

      # split Whh contributions
      ih, jh, fh, oh = tf.split(hh, 4, 1)
      ih = self.hyper_norm(ih, 'hyper_ih', use_bias=True)
      jh = self.hyper_norm(jh, 'hyper_jh', use_bias=True)
      fh = self.hyper_norm(fh, 'hyper_fh', use_bias=True)
      oh = self.hyper_norm(oh, 'hyper_oh', use_bias=True)

      # split bias
      ib, jb, fb, ob = tf.split(bias, 4, 0)  # bias is to be broadcasted.

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i = ix + ih + ib
      j = jx + jh + jb
      f = fx + fh + fb
      o = ox + oh + ob

      if self.use_layer_norm:
        concat = tf.concat([i, j, f, o], 1)
        concat = layer_norm_all(concat, batch_size, 4, self.num_units, 'ln_all')
        i, j, f, o = tf.split(concat, 4, 1)

      if self.use_recurrent_dropout:
        g = tf.nn.dropout(tf.tanh(j), self.dropout_keep_prob)
      else:
        g = tf.tanh(j)

      new_c = c * tf.sigmoid(f + self.forget_bias) + tf.sigmoid(i) * g
      new_h = tf.tanh(layer_norm(new_c, self.num_units, 'ln_c')) * tf.sigmoid(o)

      hyper_h, hyper_c = tf.split(hyper_new_state, 2, 1)
      new_total_h = tf.concat([new_h, hyper_h], 1)
      new_total_c = tf.concat([new_c, hyper_c], 1)
      new_total_state = tf.concat([new_total_h, new_total_c], 1)
    return new_h, new_total_state