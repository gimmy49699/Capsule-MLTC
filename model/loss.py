import tensorflow as tf

epsilon = 1e-9

def margin_loss(y, preds):
    # Margin Loss Function
    loss = y * tf.square(tf.maximum(0., 0.9 - preds)) \
        + 0.5 * (1.0 - y) * tf.square(tf.maximum(0., preds - 0.1))
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1), name='Margin_Loss')
    return loss


def cross_entropy_loss(y, preds):
    # Binary Cross Entropy
    loss = - (1 - y) * tf.math.log(1 - preds + epsilon) - y * tf.math.log(preds + epsilon)
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1), name='cross_entropy_loss')
    return loss

def margin_loss_v1(y, preds):
    shape = y.shape[1].value
    TP = tf.multiply(y, preds)
    FN = tf.subtract(y, TP)
    FP = preds + FN - y

    beta1 = tf.reduce_sum(FP, axis=1) / tf.reduce_sum(preds, axis=1)
    beta2 = tf.reduce_sum(FN, axis=1) / tf.reduce_sum(y, axis=1)

    reg1 = tf.tile(tf.expand_dims(beta1, axis=1), [1, shape]) * (FP * (-tf.math.log(1 - preds + epsilon)))
    reg1 = tf.reduce_mean(tf.reduce_sum(reg1, axis=1))
    reg2 = tf.tile(tf.expand_dims(beta2, axis=1), [1, shape]) * (FN * (-tf.math.log(preds + epsilon)))
    reg2 = tf.reduce_mean(tf.reduce_sum(reg2, axis=1))

    mgloss = margin_loss(y, preds)
    regloss = reg1 + reg2
    loss = mgloss + 0.1 * regloss

    return loss