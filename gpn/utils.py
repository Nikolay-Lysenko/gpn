"""
This module contains auxiliary tools.

Author: Nikolay Lysenko
"""


import os

import numpy as np
from PIL import Image


def shuffle_multiple_arrays(*arrays: np.ndarray) -> None:
    """
    Shuffle all arrays in the same manner.

    :param arrays:
        list of arrays to be shuffled
    :return:
        None
    """
    random_state = np.random.get_state()
    for array in arrays:
        np.random.set_state(random_state)
        np.random.shuffle(array)


def save_generated_image(image: np.ndarray, postfix: str) -> None:
    """
    Save generated image to a JPG file.

    :param image:
        array of shape (x_dim, y_dim, n_channels)
    :param postfix:
        name ending for a file to be created
    :return:
        None
    """
    image = 256 * image
    image = np.uint8(np.around(image))  # `PIL` supports only 8-bit data.
    if image.shape[2] == 1:  # It means that there is a monochrome image.
        img = Image.fromarray(image[:, :, 0], 'L')
    else:  # It means that there is an image in RGB system.
        img = Image.fromarray(image, 'RGB')
    img_dir = os.path.join(os.path.dirname(__file__), 'g_images')
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    img.save(os.path.join(img_dir, f'image_{postfix}.jpg'))
