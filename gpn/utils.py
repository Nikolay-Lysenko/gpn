"""
This module contains auxiliary tools.

Author: Nikolay Lysenko
"""


import numpy as np


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
