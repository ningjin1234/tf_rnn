import tensorflow as tf
import numpy as np
import unittest

# from tensorflow.python.framework import ops
# from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
# from tensorflow.python.ops import clip_ops
# from tensorflow.python.ops import embedding_ops
# from tensorflow.python.ops import init_ops
# from tensorflow.python.ops import math_ops
# from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell import _linear

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

class TkdlGRUCell(tf.nn.rnn_cell.RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self, num_units, input_size=None, activation=tanh):
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._activation = activation

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"
            with vs.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                r, u = array_ops.split(1, 2, _linear([inputs, state],
                                                     2 * self._num_units, True, 0.0))
                r, u = sigmoid(r), sigmoid(u)
            with vs.variable_scope("Candidate1"):
                c1 = _linear([inputs], self._num_units, True, 0.0)
            with vs.variable_scope("Candidate2"):
                c2 = r * _linear([state], self._num_units, False) # only needs one bias for c
            c = self._activation(c1 + c2)
            new_h = u * state + (1 - u) * c
        return new_h, new_h
