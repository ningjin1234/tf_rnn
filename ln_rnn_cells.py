from tensorflow.python.util import nest
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
import tensorflow as tf

def layer_normalization(y, g, b, epsilon=1e-6):
    mean,variance = tf.nn.moments(y, axes=[1], keep_dims=True)
    res = (y - mean) / tf.sqrt(variance+epsilon)
    return res*g + b

def init_ones():
    return init_ops.constant_initializer(1, dtype=tf.float32)

def init_zeros():
    return init_ops.constant_initializer(0, dtype=tf.float32)

class LNGRUCell(rnn_cell_impl._RNNCell):
    def __init__(self, num_units, input_size=None, activation=tanh):
        if input_size is not None:
            print("%s: The input_size parameter is deprecated." % self)
        self._num_units = num_units
        self._activation = activation

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        dim = self._num_units
        with variable_scope.variable_scope(scope or type(self).__name__):  # "GRUCell"
            with variable_scope.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                with variable_scope.variable_scope( "Layer_Parameters"):
        
                    a1 = variable_scope.get_variable("a1", [2*dim], initializer=init_ones(), dtype=tf.float32)
                    a2 = variable_scope.get_variable("a2", [2*dim], initializer=init_ones(), dtype=tf.float32)
                    a3 = variable_scope.get_variable("a3", [dim], initializer=init_ones(), dtype=tf.float32)
                    a4 = variable_scope.get_variable("a4", [dim], initializer=init_ones(), dtype=tf.float32)
                    b1 = variable_scope.get_variable("b1", [2*dim], initializer=init_zeros(), dtype=tf.float32)
                    b2 = variable_scope.get_variable("b2", [2*dim], initializer=init_zeros(), dtype=tf.float32)
                    b3 = variable_scope.get_variable("b3", [dim], initializer=init_zeros(), dtype=tf.float32)
                    b4 = variable_scope.get_variable("b4", [dim], initializer=init_zeros(), dtype=tf.float32)
        
                ln_input = _linear([inputs], 2*dim, False, scope='ln_input')
                ln_input = layer_normalization(ln_input, a1, b1)
                ln_state = _linear([state], 2*dim, False, scope='ln_state')
                ln_state = layer_normalization(ln_state, a2, b2)
                reset_update = tf.add(ln_input, ln_state)
                reset, update = array_ops.split(reset_update, [dim, dim], 1)
                reset, update = sigmoid(reset), sigmoid(update)
      
            with variable_scope.variable_scope("Candidate"):
                ln_c_input = _linear([inputs], dim, False, scope='ln_c_input')
                ln_c_input = layer_normalization(ln_c_input, a3, b3)
                ln_c_state = _linear([state], dim, False, scope='ln_c_state')
                ln_c_state = layer_normalization(ln_c_state, a4, b4)
                candidate = tf.add(ln_c_input, reset*ln_c_state)
                candidate = self._activation(candidate)
            new_state = update * candidate + (1 - update) * state
        return new_state, new_state

def _linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = scope or 'Linear'
  with variable_scope.variable_scope(scope) as outer_scope:
    weights = variable_scope.get_variable('weights', [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with variable_scope.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      biases = variable_scope.get_variable(
          'bias', [output_size],
          dtype=dtype,
          initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
    return nn_ops.bias_add(res, biases)
