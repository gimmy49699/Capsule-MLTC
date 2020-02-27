import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers


def _variable(name, shape=None, dtype=None, initializer=None,
              regularizer=None, trainable=True):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=initializer,
                              regularizer=regularizer, trainable=trainable)
    return var


def _weights(name, shape, dtype=tf.float32,
             initializer=initializers.xavier_initializer()):
    weights = _variable(name=name, shape=shape, dtype=dtype, initializer=initializer)
    return weights


def _bias(name, shape, dtype=tf.float32,
          initializer=tf.constant_initializer(0.0)):
    bias = _variable(name=name, shape=shape, dtype=dtype, initializer=initializer)
    return bias


def _conv2d(inputs, shape, strides, padding, add_bias, activation_fn=None, name=None):
    with tf.variable_scope(name):
        kernel = _weights(name="weights", shape=shape)
        output = tf.nn.conv2d(inputs, filter=kernel, strides=strides, padding=padding, name="conv2d")

        if add_bias:
            bias = _bias(name="bias", shape=[shape[-1]])
            output = tf.add(output, bias, name="conv2d")

        if activation_fn is not None:
            output = activation_fn(output, name="conv2d")
    return output


def _get_uhat(inputs, input_capsule_dim, input_capsule_num, output_capsule_dim, output_capsule_num):
    kernel = _weights(name='weights', shape=[1, input_capsule_dim, output_capsule_dim*output_capsule_num])
    u_hat = tf.keras.backend.conv1d(inputs, kernel)
    u_hat = tf.reshape(u_hat, (-1, input_capsule_num, output_capsule_num, output_capsule_dim))
    u_hat = tf.keras.backend.permute_dimensions(u_hat, (0, 2, 1, 3))
    return u_hat