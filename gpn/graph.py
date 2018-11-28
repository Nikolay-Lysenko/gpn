"""
Define computational graph and provide a session to it.

Author: Nikolay Lysenko
"""


from contextlib import contextmanager
from functools import partial
from typing import Tuple, Dict, Callable, Any, Optional

import tensorflow as tf

from gpn import discriminator_models as d_models, generator_models as g_models


def create_graph(
        d_network_fn: Callable, g_network_fn: Callable,
        d_train_op_fn: Callable, g_train_op_fn: Callable,
        d_input_shape: Tuple[Optional[int], ...],
        g_input_shape: Tuple[Optional[int], ...]
) -> tf.Graph:
    """
    Create graph with discriminator and generator.

    :param d_network_fn:
        function that transforms discriminator inputs to logits
    :param g_network_fn:
        function that transforms noise to generated images
    :param d_train_op_fn:
        function that returns discriminator training operation
    :param g_train_op_fn:
        function that returns generator training operation
    :param d_input_shape:
        shape of discriminator input
    :param g_input_shape:
        shape of generator input
    :return:
        computation graph
    """
    graph = tf.Graph()
    with graph.as_default():
        # Discriminator training.
        d_input = tf.placeholder(tf.float32, d_input_shape, name='d_input')
        d_labels = tf.placeholder(tf.int32, name='d_labels')
        d_logits = d_network_fn(d_input, reuse=False)
        d_train_op = d_train_op_fn(d_logits, d_labels)

        # Predicting with discriminator.
        d_pred_logits = d_network_fn(d_input, reuse=True)
        d_preds = tf.argmax(input=d_pred_logits, axis=1, name='d_predictions')

        # Generator training.
        g_input = tf.placeholder(tf.float32, g_input_shape)
        # TODO: Implement.
        # g_pictures = g_network_fn(g_input)
        # g_padded_pictures = tf.pad(g_pictures)
        # g_parts = ...
        # g_neg_loss, _ = d_network_fn(g_parts, reuse=True)
        # g_loss = tf.negative(g_neg_loss)  # TODO: Softmax?
        # g_train_op = g_train_op_fn(g_loss)

    return graph


def get_networks(settings: Dict[str, Any]) -> Tuple[Callable, Callable]:
    """
    Return two functions that define structures of neural networks.

    The first function transforms discriminator inputs to logits
    and predicted labels. The second function transforms random noise
    to images.

    :param settings:
        configuration of an experiment
    :return:
        functions that define structures of
        a discriminator network and a generator network
    """
    dispatcher = {
        'basic_mnist_d_network': d_models.basic_mnist_d_network_fn,
        'basic_mnist_g_network': g_models.basic_mnist_g_network_fn
    }
    d_network_fn = dispatcher[settings['discriminator']['network']]
    g_network_fn = dispatcher[settings['generator']['network']]
    return d_network_fn, g_network_fn


def get_train_ops(settings: Dict[str, Any]) -> Tuple[Callable, Callable]:
    """
    Return two functions that define training operations for models.

    :param settings:
        configuration of an experiment
    :return:
        functions that define training operations for
        discriminator and generator
    """
    d_train_settings = settings['discriminator']['training']
    dispatcher = {
        'basic_mnist_d_train_op': partial(
            d_models.basic_mnist_d_train_op_fn,
            learning_rate=d_train_settings['learning_rate'],
            beta_one=d_train_settings['beta_one']
        ),
        'basic_mnist_g_train_op': g_models.basic_mnist_g_train_op_fn
    }
    d_train_op_fn = dispatcher[settings['discriminator']['train_op']]
    g_train_op_fn = dispatcher[settings['generator']['train_op']]
    return d_train_op_fn, g_train_op_fn


def calculate_input_shapes(
        settings: Dict[str, Any]
) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
    """
    Calculate shapes of inputs based on settings of experiment.

    :param settings:
        configuration of an experiment
    :return:
        shapes of discriminator input and generator input
    """
    data_shape = settings['data']['shape']
    n_channels = data_shape[0]
    internal_size = settings['discriminator']['setup']['internal_size']
    frame_size = settings['discriminator']['setup']['frame_size']
    fragment_size = internal_size + 2 * frame_size
    d_input_shape = (None, fragment_size, fragment_size, n_channels)
    g_input_shape = (None, settings['generator']['setup']['z_dim'])
    return d_input_shape, g_input_shape


@contextmanager
def create_session(settings: Dict[str, Any]) -> tf.Session:
    """
    Create a context with session to the graph.

    :param settings:
        configuration of an experiment
    :return:
        session
    """
    d_network_fn, g_network_fn = get_networks(settings)
    d_train_op_fn, g_train_op_fn = get_train_ops(settings)
    d_input_shape, g_input_shape = calculate_input_shapes(settings)
    graph = create_graph(
        d_network_fn, g_network_fn,
        d_train_op_fn, g_train_op_fn,
        d_input_shape, g_input_shape
    )
    with graph.as_default():
        session = tf.Session()
        yield session
