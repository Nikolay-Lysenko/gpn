"""
Define model functions for generators.

Author: Nikolay Lysenko
"""


import tensorflow as tf


def basic_mnist_g_network_fn(g_input: tf.Tensor, reuse: bool) -> tf.Tensor:
    """
    Define network structure for a basic MNIST-related discriminator.

    :param g_input:
        tensor to be passed to the generator as input
    :param reuse:
        if `False`, all variables are created from scratch,
        if `True`, variables that already exist are used
    :return:
        images that are drawn by the generator,
        their shape is (n_images, x_dim, y_dim, n_channels),
        values are in a range from 0 to 1
    """
    with tf.variable_scope('generator', reuse=reuse):
        first_dense_layer = tf.layers.dense(
            inputs=g_input,
            units=7*7*32,
            activation=tf.nn.relu
        )
        first_dense_layer_reshaped = tf.reshape(
            tensor=first_dense_layer,
            shape=(-1, 7, 7, 32)
        )
        first_transposed_conv_layer = tf.layers.conv2d_transpose(
            inputs=first_dense_layer_reshaped,
            filters=16,
            kernel_size=(3, 3),
            strides=[2, 2],
            padding='same'
        )
        second_transposed_conv_layer = tf.layers.conv2d_transpose(
            inputs=first_transposed_conv_layer,
            filters=1,
            kernel_size=[3, 3],
            strides=[2, 2],
            padding='same'
        )
        images = tf.tanh(second_transposed_conv_layer)
        images = tf.identity(images, name='images')
        return images


def basic_mnist_g_train_op_fn(
        loss: tf.Tensor, learning_rate: float, beta_one: float
) -> tf.Operation:
    """
    Define training operation for basic MNIST-related generator.

    :param loss:
        plausibility of images estimated by the discriminator
    :param learning_rate:
        learning rate for Adam optimizer
    :param beta_one:
        exponential decay rate for the first moment estimates
    :return:
        training operation
    """
    # Scope is needed, because `AdamOptimizer` is also used by discriminator.
    # Thus, there are name clashes for internal variables of `AdamOptimizer`
    # without a scope.
    with tf.variable_scope('g_op', reuse=False):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=beta_one
        )
        training_operation = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step(),
            name='g_train_op'
        )
        return training_operation
