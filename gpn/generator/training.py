"""
Train generator.

Author: Nikolay Lysenko
"""


import os
from typing import Dict, Any

import numpy as np
import tensorflow as tf
from numpy.random import choice

from gpn.graph import create_session
from gpn.utils import save_generated_image


def yield_batches(num_batches: int, batch_size: int, z_dim: int) -> np.ndarray:
    """
    Generate noise from standard normal distribution.

    :param num_batches:
        number of batches to generate
    :param batch_size:
        number of objects per batch
    :param z_dim:
        dimensionality of noise
    :yield:
        batches with noise
    """
    for i in range(num_batches):
        yield np.random.normal(size=(batch_size, z_dim))


def train(settings: Dict[str, Any]) -> None:
    """
    Train generator.

    :param settings:
        configuration of an experiment
    :return:
        None
    """
    g_train_settings = settings['generator']['training']
    num_epochs = g_train_settings['num_epochs']
    batch_size = g_train_settings['batch_size']
    g_setup = settings['generator']['setup']
    num_batches = g_setup['num_batches']
    z_dim = g_setup['z_dim']
    d_setup = settings['discriminator']['setup']
    internal_size = d_setup['internal_size']

    with create_session(settings) as sess:
        saver = tf.train.Saver()
        checkpoint_path = os.path.join(
            os.path.dirname(__file__),
            settings['discriminator']['saving_path']
        )
        saver.restore(sess, checkpoint_path)

        g_input = sess.graph.get_tensor_by_name('g_input:0')
        g_images = sess.graph.get_tensor_by_name('generator/images:0')
        g_corners = [
            sess.graph.get_tensor_by_name(f'g_corner_{i}:0')
            for i in range(g_setup['n_fragments_per_image'])
        ]
        g_train_op = sess.graph.get_operation_by_name('g_op/g_train_op')
        g_loss = sess.graph.get_tensor_by_name('g_loss:0')

        n_batches_passed = 0
        for epoch_i in range(num_epochs):
            batches = yield_batches(num_batches, batch_size, z_dim)
            for batch_noise in batches:
                n_batches_passed += 1

                n_x_options = settings['data']['shape'][1] - internal_size + 1
                n_y_options = settings['data']['shape'][2] - internal_size + 1
                g_corners_with_values = {
                    g_corner: (choice(n_x_options), choice(n_y_options))
                    for g_corner in g_corners
                }

                feed_dict = {g_input: batch_noise, **g_corners_with_values}
                sess.run(g_train_op, feed_dict)

                if n_batches_passed % 1000 == 0:
                    batch_loss = g_loss.eval(feed_dict, sess)
                    print(f'Epoch {epoch_i}: loss on a batch = {batch_loss}')
                    image = g_images.eval(feed_dict, sess)[0]
                    save_generated_image(image, n_batches_passed)

        saving_path = os.path.join(
            os.path.dirname(__file__),
            settings['generator']['saving_path']
        )
        saver.save(sess, saving_path)
