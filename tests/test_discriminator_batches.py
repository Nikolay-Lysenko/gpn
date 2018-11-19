"""
Test `discriminator_batches.py` module.

Author: Nikolay Lysenko
"""


from typing import List

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


@pytest.mark.parametrize(
    "image_arr, internal_size, padding_size, possible_outputs",
    [
        (
            np.array([
                [[0, 1],
                 [1, 0]],
                [[1, 1],
                 [0, 0]],
            ]),
            1,
            1,
            [
                np.array([
                    [[0, 0, 0],
                     [0, 0, 1],
                     [0, 1, 0]],
                    [[0, 0, 0],
                     [0, 1, 1],
                     [0, 0, 0]]
                ]),
                np.array([
                    [[0, 0, 0],
                     [0, 1, 0],
                     [1, 0, 0]],
                    [[0, 0, 0],
                     [1, 1, 0],
                     [0, 0, 0]]
                ]),
                np.array([
                    [[0, 0, 1],
                     [0, 1, 0],
                     [0, 0, 0]],
                    [[0, 1, 1],
                     [0, 0, 0],
                     [0, 0, 0]]
                ]),
                np.array([
                    [[0, 1, 0],
                     [1, 0, 0],
                     [0, 0, 0]],
                    [[1, 1, 0],
                     [0, 0, 0],
                     [0, 0, 0]]
                ])
            ]
        )
    ]
)
def test_sample_square_part(
        image_arr: np.ndarray, internal_size: int, padding_size: int,
        possible_outputs: List[np.ndarray]
) -> None:
    """Test `sample_square_part` function."""
    output = batches.sample_square_part(image_arr, internal_size, padding_size)
    print(output)
    assert any([np.array_equal(output, arr) for arr in possible_outputs])
