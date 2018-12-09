"""
Define computational graph and provide a session to it.

Author: Nikolay Lysenko
"""


from contextlib import contextmanager
from functools import partial
from typing import List, Tuple, Dict, Callable, Any, Optional

import tensorflow as tf

from gpn import discriminator_models as d_models, generator_models as g_models


def sample_multiple_fragments(
        images: tf.Tensor, corners: List[tf.Tensor],
        fragment_size: int, frame_size: int, n_channels: int
) -> tf.Tensor:
    """
    Sample fragments of images.

    :param images:
        images to be used for sampling their fragments
    :param corners:
        placeholders for left bottom corners of fragments;
        length of the parameter defines number of fragments
        to be sampled from an image
    :param fragment_size:
        full size of a fragment
    :param frame_size:
        size of an outer part of fragment that can be outside of
        an original image
    :param n_channels:
        number of channels in images
    :return:
        fragments that can be passed to discriminator
    """
    paddings = tf.constant([
        [0, 0],
        [frame_size, frame_size],
        [frame_size, frame_size],
        [0, 0]
    ])
    padded_images = tf.pad(images, paddings)
    all_fragments = []
    for corner in corners:
        corner_fragments = padded_images[
            :,
            corner[0]:(corner[0] + fragment_size),
            corner[1]:(corner[1] + fragment_size),
            :
        ]
        # Non-tensor dimensions are required further.
        fragment_shape = (-1, fragment_size, fragment_size, n_channels)
        corner_fragments = tf.reshape(corner_fragments, fragment_shape)
        all_fragments.append(corner_fragments)
    fragments = tf.concat(all_fragments, axis=0)
    return fragments


def create_graph(
        d_network_fn: Callable, g_network_fn: Callable,
        d_train_op_fn: Callable, g_train_op_fn: Callable,
        fragments_sampling_fn: Callable, n_fragments_per_g_image: int,
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
    :param fragments_sampling_fn:
        function that samples fragments of generated images
    :param n_fragments_per_g_image:
        number of fragments to sample from each generated image;
        such fragments are passed to discriminator in order to
        calculate generator loss
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

        # Predicting with discriminator (needed for its evaluation).
        d_pred_logits = d_network_fn(d_input, reuse=True)
        d_preds = tf.argmax(input=d_pred_logits, axis=1, name='d_predictions')

        # Generator training.
        g_input = tf.placeholder(tf.float32, g_input_shape, name='g_input')
        g_images = g_network_fn(g_input, reuse=False)
        corners = [
            tf.placeholder(tf.int32, (2,), name=f'g_corner_{i}')
            for i in range(n_fragments_per_g_image)
        ]
        g_fragments = fragments_sampling_fn(g_images, corners)
        g_logits = d_network_fn(g_fragments, reuse=True)
        g_loss = tf.reduce_sum(tf.sigmoid(g_logits)[:, 0])
        g_loss = tf.identity(g_loss, name='g_loss')
        g_train_op = g_train_op_fn(g_loss)

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
    g_train_settings = settings['generator']['training']
    dispatcher = {
        'basic_mnist_d_train_op': partial(
            d_models.basic_mnist_d_train_op_fn,
            learning_rate=d_train_settings['learning_rate'],
            beta_one=d_train_settings['beta_one']
        ),
        'basic_mnist_g_train_op': partial(
            g_models.basic_mnist_g_train_op_fn,
            learning_rate=g_train_settings['learning_rate'],
            beta_one=g_train_settings['beta_one']
        )
    }
    d_train_op_fn = dispatcher[settings['discriminator']['train_op']]
    g_train_op_fn = dispatcher[settings['generator']['train_op']]
    return d_train_op_fn, g_train_op_fn


def get_fragments_sampling_fn(settings: Dict[str, Any]) -> Callable:
    """
    Return function that samples fragments of generated images.

    :param settings:
        configuration of an experiment
    :return:
        function that samples fragments of generated images
    """
    d_setup = settings['discriminator']['setup']
    frame_size = d_setup['frame_size']
    fragment_size = d_setup['internal_size'] + 2 * frame_size
    fragments_sampling_fn = partial(
        sample_multiple_fragments,
        frame_size=frame_size,
        fragment_size=fragment_size,
        n_channels=settings['data']['shape'][0]
    )
    return fragments_sampling_fn


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
    fragment_sampling_fn = get_fragments_sampling_fn(settings)
    d_input_shape, g_input_shape = calculate_input_shapes(settings)
    g_setup = settings['generator']['setup']
    n_fragments_per_g_image = g_setup['n_fragments_per_image']
    graph = create_graph(
        d_network_fn, g_network_fn,
        d_train_op_fn, g_train_op_fn,
        fragment_sampling_fn, n_fragments_per_g_image,
        d_input_shape, g_input_shape
    )
    with graph.as_default():
        session = tf.Session()
        yield session
