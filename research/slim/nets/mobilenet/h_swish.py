# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

def h_swish(net):
  """ hard swish activation funcsion.

  Architecture: https://arxiv.org/pdf/1905.02244.pdf, and Semantic Segmentation of Satellite Images using a Modified CNN with
Hard-Swish Activation Function

  Args:
    features: A Tensor with the type float, double, int32, int64, uint8, int16 or int8.

  Returns:
    A Tensor with the same type as features.
  """
  with tf.variable_scope('h_swish'):
    return tf.multiply(net, tf.divide(tf.nn.relu6(net + 3), 6))