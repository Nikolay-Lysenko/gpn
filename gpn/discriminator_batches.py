"""
Prepare batches for discriminator.

Author: Nikolay Lysenko
"""


import numpy as np


def pad_image(image_arr: np.ndarray, padding_size: int) -> np.ndarray:
    """
    Create padding with zeros for all spatial dimensions.

    :param image_arr:
        array of shape (channel_dim, x_dim, y_dim)
    :param padding_size:
        number of pixels to create near each border
    :return:
        padded array
    """
    n_channels = image_arr.shape[0]
    initial_width = image_arr.shape[2]
    vertical_padding = np.zeros((n_channels, padding_size, initial_width))
    arrays = (vertical_padding, image_arr, vertical_padding)
    image_arr = np.concatenate(arrays, axis=1)
    new_height = image_arr.shape[1]
    horizontal_padding = np.zeros((n_channels, new_height, padding_size))
    arrays = (horizontal_padding, image_arr, horizontal_padding)
    image_arr = np.concatenate(arrays, axis=2)
    return image_arr
