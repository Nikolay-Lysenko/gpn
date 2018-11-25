"""
Define model functions for discriminators.

Such functions can take more arguments than its supposed by
`tf.Estimator`, so extra arguments must be filled with
`functools.partial`.

Author: Nikolay Lysenko
"""


from typing import Dict

import tensorflow as tf


def basic_mnist_model_fn(
        features: Dict[str, tf.Tensor],
        labels: tf.Tensor,
        mode: str,
        learning_rate: float,
        beta_one: float
) -> tf.estimator.EstimatorSpec:
    """
    Define model function for a basic MNIST-related discriminator.

    Here, parameters of layers are selected for the following setup:
    * n_fragments_per_image = 7,
    * internal_size = 3,
    * frame_size = 1.
    If another setup is studied, another model function should be
    defined and used.

    :param features:
        dictionary with a key 'data' and a tensor of shape
        (None, fragment_size, fragment_size, 1) as its value
    :param labels:
        tensor with binary labels
    :param mode:
        key that indicates a goal of the function call
        (to train, to evaluate, or to predict)
    :param learning_rate:
        learning rate for Adam optimizer
    :param beta_one:
        exponential decay rate for the first moment estimates
    :return:
        estimator specification for a particular mode
    """
    # Define architecture.
    input_layer = features['data']
    first_conv_layer = tf.layers.conv2d(
        inputs=input_layer,
        filters=16,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )
    first_pooling_layer = tf.layers.max_pooling2d(
        inputs=first_conv_layer,
        pool_size=[2, 2],
        strides=2,
        padding='same',
    )
    second_conv_layer = tf.layers.conv2d(
        inputs=first_pooling_layer,
        filters=32,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )
    flat_second_conv_layer = tf.layers.flatten(
        second_conv_layer
    )
    dense_layer = tf.layers.dense(
        inputs=flat_second_conv_layer,
        units=128,
        activation=tf.nn.relu
    )
    dropout_layer = tf.layers.dropout(
        inputs=dense_layer,
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )
    logits = tf.layers.dense(
        inputs=dropout_layer,
        units=2
    )
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="pred_proba")
    }

    # Output specifications.
    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions
        )
        return spec
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=beta_one
        )
        training_operation = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=training_operation
        )
        return spec
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels,
                predictions=predictions["classes"]
            )
        }
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops
        )
        return spec
    else:
        raise ValueError(f'Unknown mode: {mode}')
