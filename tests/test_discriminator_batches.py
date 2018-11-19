"""
Test `discriminator_batches.py` module.

Author: Nikolay Lysenko
"""


import pytest
import numpy as np

from gpn import discriminator_batches as batches


@pytest.mark.parametrize(
    "image_arr, padding_size, expected",
    [
        (
            np.array([
                [[0, 1, 0],
                 [1, 0, 1]],
                [[1, 0, 1],
                 [0, 1, 0]]
            ]),
            1,
            np.array([
                [[0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0],
                 [0, 1, 0, 1, 0],
                 [0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0],
                 [0, 1, 0, 1, 0],
                 [0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0]]
            ])
        )
    ]
)
def test_pad_image(
        image_arr: np.ndarray, padding_size: int, expected: np.ndarray
) -> None:
    """Test `pad_image` function."""
    padded_image_arr = batches.pad_image(image_arr, padding_size)
    np.testing.assert_equal(padded_image_arr, expected)
