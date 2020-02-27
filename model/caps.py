import tensorflow as tf
from keras import backend as K
from model.utils import _get_uhat, _conv2d

epsilon = 1e-9


# define Squash function
def squash(x, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(x), axis, keepdims=True) + epsilon
    scale = tf.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x

# define routing process
def routing(u_hat, iter_times, i_activations, amendment, leakysoftmax):
    b = tf.zeros_like(u_hat[:, :, :, 0])
    output_caps_num = b.shape[1].value

    i_activations = tf.expand_dims(i_activations, axis=-2)
    i_activations = tf.tile(i_activations, [1, output_caps_num, 1])

    for i in range(iter_times):
        if leakysoftmax:
            if amendment:
                b = tf.multiply(i_activations, b)
            leak = tf.reduce_sum(tf.zeros_like(b), axis=1, keepdims=True)
            leaky_logits = tf.concat([leak, b], axis=1)
            leaky_cij = tf.nn.softmax(leaky_logits, axis=1)
            c = tf.split(leaky_cij, [1, output_caps_num], axis=1, name='leak')[1]
        else:
            if amendment:
                b = tf.multiply(i_activations, b)
            c = tf.nn.softmax(b, axis=1)
        output = squash(K.batch_dot(c, u_hat, [2, 2]))
        if i < iter_times - 1:
            b = b + K.batch_dot(output, u_hat, [2, 3])
    caps = output
    activations = K.sqrt(K.sum(K.square(caps), 2))
    return caps, activations


def PrimaryCaps(inputs, num_out_caps, out_caps_shape, training,
                strides=[1, 1, 1, 1], padding="VALID", add_bias=True):
    input_nums, input_dims = inputs.shape[1], inputs.shape[-1]
    with tf.variable_scope("PrimaryCaps"):
        inputs = tf.layers.batch_normalization(inputs, training=training)
        caps = _conv2d(inputs, shape=[1, 1, input_dims, num_out_caps*out_caps_shape],
                       strides=strides, padding=padding, add_bias=add_bias, name="poses")
        caps = tf.reshape(caps, [-1, input_nums, num_out_caps, out_caps_shape])
        caps = squash(caps)
        activations = K.sqrt(K.sum(K.square(caps), axis=-1))
        caps = tf.reshape(caps, [-1, input_nums*num_out_caps, out_caps_shape], name="capsules")
        activations = tf.reshape(activations, [-1, input_nums*num_out_caps], name="activations")
    return caps, activations


def FullyConnectCaps(inputs, activations, training, num_out_caps, out_caps_shape, iter_times,
                     amendment, leakysoftmax, name=None):
    with tf.variable_scope(name):
        inputs = tf.layers.batch_normalization(inputs, training=training)
        _, num_in, in_dims = inputs.get_shape().as_list()

        u_hat = _get_uhat(inputs, input_capsule_dim=in_dims, input_capsule_num=num_in,
                          output_capsule_dim=out_caps_shape, output_capsule_num=num_out_caps)

        caps, activations = routing(u_hat, iter_times, activations, amendment, leakysoftmax)
    return caps, activations