"""
Define model functions for discriminators.

Author: Nikolay Lysenko
"""


import tensorflow as tf


def basic_mnist_d_network_fn(d_input: tf.Tensor, reuse: bool) -> tf.Tensor:
    """
    Define network structure for a basic MNIST-related discriminator.

    Here, parameters of layers are selected for the following setup:
    * n_fragments_per_image = 7,
    * internal_size = 2,
    * frame_size = 1.
    If another setup is studied, another function should be
    defined and used.

    :param d_input:
        tensor to be passed to the discriminator as input
    :param reuse:
        if `False`, all variables are created from scratch,
        if `True`, variables that already exist are used
    :return:
        logits that are calculated by discriminator
    """
    with tf.variable_scope('discriminator', reuse=reuse):
        first_conv_layer = tf.layers.conv2d(
            inputs=d_input,
            filters=16,
            kernel_size=[2, 2],
            activation=tf.nn.relu
        )
        second_conv_layer = tf.layers.conv2d(
            inputs=first_conv_layer,
            filters=32,
            kernel_size=[2, 2],
            activation=tf.nn.relu
        )
        flat_layer = tf.layers.flatten(
            second_conv_layer
        )
        dense_layer = tf.layers.dense(
            inputs=flat_layer,
            units=64,
            activation=tf.nn.relu
        )
        dropout_layer = tf.layers.dropout(
            inputs=dense_layer,
            rate=0.4,
            training=(not reuse)
        )
        logits = tf.layers.dense(
            inputs=dropout_layer,
            units=2
        )
        return logits


def basic_mnist_d_train_op_fn(
        logits: tf.Tensor, labels: tf.Tensor,
        learning_rate: float, beta_one: float
) -> tf.Operation:
    """
    Define training operation for basic MNIST-related discriminator.

    :param logits:
        logits that are calculated by discriminator
    :param labels:
        true labels
    :param learning_rate:
        learning rate for Adam optimizer
    :param beta_one:
        exponential decay rate for the first moment estimates
    :return:
        training operation
    """
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    loss = tf.identity(loss, name='d_loss')
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=beta_one
    )
    training_operation = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step(),
        name='d_train_op'
    )
    return training_operation
