"""
Train discriminator.

Author: Nikolay Lysenko
"""


import os
from typing import Tuple, Dict, Callable, Generator, Any

import numpy as np
import tensorflow as tf

from gpn.graph import create_session
from gpn.discriminator_dataset import generate_dataset


def get_mnist_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Get images from MNIST dataset.

    :return:
        train set and test set
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    max_intensity = 256
    x_train = x_train.reshape(
        x_train.shape[0], 1, x_train.shape[1], x_train.shape[2]
    )
    x_train = x_train / max_intensity
    x_test = x_test.reshape(
        x_test.shape[0], 1, x_test.shape[1], x_test.shape[2]
    )
    x_test = x_test / max_intensity
    return x_train, x_test


def get_dataset_loader_by_name(name: str) -> Callable:
    """
    Get dataset loader by its name.

    :param name:
        name of dataset to be loaded
    :return:
        function that loads dataset
    """
    dispatcher = {
        'mnist': get_mnist_data
    }
    dataset_downloader = dispatcher[name]
    return dataset_downloader


def yield_batches(
        data: np.ndarray, labels: np.ndarray, batch_size: int
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Split train data into batches and iterate over them.

    :param data:
        feature representation of images,
        shape is (n_images, x_dim, y_dim, n_channels)
    :param labels:
        labels of objects
    :param batch_size:
        number of objects per batch
    :yield:
        batches of initial dataset
    """
    assert data.shape[0] == labels.shape[0]
    size_of_incomplete_batch = data.shape[0] % batch_size
    if size_of_incomplete_batch > 0:
        data = data[:-size_of_incomplete_batch, :, :, :]
        labels = labels[:-size_of_incomplete_batch, :, :, :]
    data = data.reshape(-1, batch_size, *data.shape[1:])
    labels = labels.reshape(-1, batch_size)
    for i in range(data.shape[0]):
        yield data[i], labels[i]


def train(settings: Dict[str, Any]) -> None:
    """
    Train discriminator.

    :param settings:
        configuration of an experiment
    :return:
        None
    """
    # Prepare data.
    loader = get_dataset_loader_by_name(settings['data']['dataset_name'])
    train_images, test_images = loader()
    setup = settings['discriminator']['setup']
    train_data, train_labels = generate_dataset(train_images, **setup)
    test_data, test_labels = generate_dataset(test_images, **setup)

    # Train discriminator.
    d_train_settings = settings['discriminator']['training']
    num_epochs = d_train_settings['num_epochs']
    batch_size = d_train_settings['batch_size']
    with create_session(settings) as sess:
        d_input = sess.graph.get_tensor_by_name('d_input:0')
        d_labels = sess.graph.get_tensor_by_name('d_labels:0')
        d_train_op = sess.graph.get_operation_by_name('d_train_op')
        d_loss = sess.graph.get_tensor_by_name('d_loss:0')
        d_predictions = sess.graph.get_tensor_by_name('d_predictions:0')
        sess.run(tf.global_variables_initializer())

        n_batches_passed = 0
        for epoch_i in range(num_epochs):
            batches = yield_batches(train_data, train_labels, batch_size)
            for batch_data, batch_labels in batches:
                n_batches_passed += 1
                feed_dict = {d_input: batch_data, d_labels: batch_labels}
                sess.run(d_train_op, feed_dict)

                if n_batches_passed % 1000 == 0:
                    batch_loss = d_loss.eval(feed_dict, sess)
                    print(f'Epoch {epoch_i}: loss on a batch = {batch_loss}')
        saver = tf.train.Saver()
        saving_path = os.path.join(
            os.path.dirname(__file__),
            settings['discriminator']['saving_path']
        )
        saver.save(sess, saving_path)

        test_predictions = d_predictions.eval({d_input: test_data}, sess)
        accuracy = (test_labels == test_predictions).mean()
        print(f'Accuracy on hold-out test set: {accuracy}')
