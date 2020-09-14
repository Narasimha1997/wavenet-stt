import tensorflow as tf
from tensorflow.python.ops import init_ops
import tensorflow.contrib.slim as slim


# def _atrous_conv1d(value, filters, rate, padding, name=None):
#  return tf.nn.convolution(input=value,
#                           filter=filters,
#                           padding=padding,
#                           dilation_rate=np.broadcast_to(rate, (1,)),
#                           name=name)


def _aconv1d(inputs,
             num_outputs=None,
             causal=False,
             kerner_size=7,
             padding='SAME',
             rate=2,  #
             activation_fn=tf.nn.tanh,
             normalizer_fn=None,
             normalizer_params=None,
             weights_initializer=None,
             biases_initializer=init_ops.zeros_initializer(),
             weights_regularizer=None,
             biases_regularizer=None,
             scope=None):
  """Functional interface for 1d atrous convolution (a.k.a. convolution with holes or dilated convolution).

  Reference:
  sugartensor/sg_layer.py sg_aconv1d at line 179
  """
  use_bias = not normalizer_fn and biases_initializer

  with tf.variable_scope(scope, default_name='aconv1d'):
    shape = inputs.get_shape().as_list()
    channels = shape[-1]
    if num_outputs is None or num_outputs <= 0:
      num_outputs = channels

    weights = tf.get_variable('weights',
                              (1, kerner_size, channels, num_outputs),
                              dtype=inputs.dtype,
                              initializer=weights_initializer,
                              regularizer=weights_regularizer)
    biases = tf.get_variable('biases', num_outputs,
                             dtype=inputs.dtype,
                             initializer=biases_initializer,
                             regularizer=biases_regularizer) if use_bias else 0

    if causal:
      if padding.lower() == 'same':
        pad_len = (kerner_size - 1) * rate
        inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
      inputs = tf.expand_dims(inputs, axis=1)
      outputs = tf.nn.atrous_conv2d(inputs, weights, rate=rate, padding='VALID') + biases
    else:
      inputs = tf.expand_dims(inputs, axis=1)
      outputs = tf.nn.atrous_conv2d(inputs, weights, rate=rate, padding=padding) + biases
    outputs = tf.squeeze(outputs, axis=1)
    if normalizer_fn is not None:
      normalizer_params = normalizer_params or {}
      outputs = normalizer_fn(outputs, **normalizer_params)
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs


def _resnet_block(inputs, num_outputs, kernel_size, rate,
                  normalizer_fn=None,
                  normalizer_params=None,
                  weights_initializer=None,
                  biases_initializer=init_ops.zeros_initializer(),
                  weights_regularizer=None,
                  biases_regularizer=None,
                  scope=None):
  with tf.variable_scope(scope, default_name="block_%d" % rate):
    conv_filter = _aconv1d(inputs,
                           kerner_size=kernel_size,
                           rate=rate,
                           activation_fn=tf.nn.tanh,
                           normalizer_fn=normalizer_fn,
                           normalizer_params=normalizer_params,
                           weights_initializer=weights_initializer,
                           biases_initializer=biases_initializer,
                           weights_regularizer=weights_regularizer,
                           biases_regularizer=biases_regularizer,
                           scope='filter')
    conv_gate = _aconv1d(inputs,
                         kerner_size=kernel_size,
                         rate=rate,
                         activation_fn=tf.nn.sigmoid,
                         normalizer_fn=normalizer_fn,
                         normalizer_params=normalizer_params,
                         weights_initializer=weights_initializer,
                         biases_initializer=biases_initializer,
                         weights_regularizer=weights_regularizer,
                         biases_regularizer=biases_regularizer,
                         scope='gate')
    outputs = conv_filter * conv_gate
    outputs = slim.conv1d(outputs, num_outputs,
                          kernel_size=1,
                          activation_fn=tf.nn.tanh,
                          normalizer_fn=normalizer_fn,
                          normalizer_params=normalizer_params,
                          weights_initializer=weights_initializer,
                          biases_initializer=biases_initializer,
                          weights_regularizer=weights_regularizer,
                          biases_regularizer=biases_regularizer,
                          scope='conv')
    return outputs + inputs, outputs


def bulid_wavenet(inputs, num_classes, is_training=False, num_hidden_size=128, num_layers=3, rates=[1, 2, 4, 8, 16],
                  scope=None):
  """
  I don't kown how to implement the he_uniform(the default initializer of orignal code)with tensorflow.
  and I use the tf.contrib.layers.xavier_initializer() as default weights_initializer.

  :param inputs:
  :param num_classes:
  :param num_hidden_size:
  :param num_layers:
  :param rates:
  :param scope:
  :return:
  """

  def get_initializer(name=None):
    if name is None:
      return tf.contrib.layers.xavier_initializer()

  def get_normalizer_params():
    return {'is_training': is_training, 'scale': True}

  outputs = 0
  with tf.variable_scope(scope, default_name='wavenet'):
    with tf.variable_scope('input'):
      nets = slim.conv1d(inputs, num_hidden_size,
                         kernel_size=1,
                         activation_fn=tf.nn.tanh,
                         normalizer_fn=slim.batch_norm,
                         normalizer_params=get_normalizer_params(),
                         weights_initializer=get_initializer(),
                         scope='conv')

    with tf.variable_scope('resnet'):
      for i in range(num_layers):
        for rate in rates:
          nets, output = _resnet_block(nets, num_hidden_size,
                                       kernel_size=7, rate=rate,
                                       normalizer_fn=slim.batch_norm,
                                       normalizer_params=get_normalizer_params(),
                                       weights_initializer=get_initializer(),
                                       scope='block_%d_%d' % (i, rate))
          outputs += output

    with tf.variable_scope('output'):
      outputs = slim.conv1d(outputs, num_hidden_size,
                            kernel_size=1,
                            activation_fn=tf.nn.tanh,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=get_normalizer_params(),
                            weights_initializer=get_initializer(),
                            scope='conv')
      return slim.conv1d(outputs,
                         num_outputs=num_classes,
                         kernel_size=1,
                         normalizer_params=get_normalizer_params(),
                         weights_initializer=get_initializer(),
                         scope='logit')
