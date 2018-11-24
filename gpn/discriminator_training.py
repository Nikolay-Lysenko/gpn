"""
Train discriminator.

Author: Nikolay Lysenko
"""


from typing import Tuple, Dict, Callable, Any
from functools import partial

import numpy as np
import tensorflow as tf

from gpn import discriminator_models as d_models
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


def get_model_fn(settings: Dict[str, Any]) -> Callable:
    """
    Get model function compatible with `tf.Estimator`.

    :param settings:
        configuration of an experiment
    :return:
        model function ready for usage
    """
    dispatcher = {
        'basic_mnist_discriminator': partial(
            d_models.basic_mnist_model_fn,
            learning_rate=settings['discriminator_training']['learning_rate'],
            beta_one=settings['discriminator_training']['beta_one']
        )
    }
    name = settings['discriminator']['model_fn']
    model_fn = dispatcher[name]
    return model_fn


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
    setup = settings['setup']
    train_data, train_labels = generate_dataset(train_images, **setup)
    test_data, test_labels = generate_dataset(test_images, **setup)

    # Create estimator.
    model_fn = get_model_fn(settings)
    model_dir = settings['discriminator']['model_dir']
    discriminator = tf.estimator.Estimator(model_fn, model_dir)

    # Train estimator.
    train_settings = settings['discriminator_training']
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'data': train_data},
        y=train_labels,
        batch_size=train_settings['batch_size'],
        num_epochs=train_settings['num_epochs'],
        shuffle=True
    )
    discriminator.train(input_fn=train_input_fn)

    # Evaluate estimator.
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'data': test_data},
        y=test_labels,
        num_epochs=1,
        shuffle=False
    )
    metrics = discriminator.evaluate(input_fn=eval_input_fn)
    print(metrics)
