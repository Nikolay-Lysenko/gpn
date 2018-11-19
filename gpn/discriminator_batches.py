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


def sample_square_part(
        image_arr: np.ndarray, internal_size: int, padding_size: int
) -> np.ndarray:
    """
    Sample a square part of an image.

    :param image_arr:
        array of shape (channel_dim, x_dim, y_dim)
    :param internal_size:
        size of square that must be entirely within the original image
    :param padding_size:
        size of a frame that can be outside of the original image
    :return:
        square image of size `frame_size + padding_size`
    """
    x_options = list(range(image_arr.shape[1] - internal_size + 1))
    x_corner = np.random.choice(x_options)
    y_options = list(range(image_arr.shape[2] - internal_size + 1))
    y_corner = np.random.choice(y_options)
    padded_image_arr = pad_image(image_arr, padding_size)
    x_min = x_corner + padding_size - 1
    x_max = x_corner + internal_size + 2 * padding_size
    y_min = y_corner + padding_size - 1
    y_max = y_corner + internal_size + 2 * padding_size
    part = padded_image_arr[:, x_min:x_max, y_min:y_max]
    return part
