"""
Test `discriminator_batches.py` module.

Author: Nikolay Lysenko
"""


from typing import List

import pytest
import numpy as np

from gpn import discriminator_batches as batches


@pytest.mark.parametrize(
    "image, padding_size, expected",
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
        image: np.ndarray, padding_size: int, expected: np.ndarray
) -> None:
    """Test `pad_image` function."""
    padded_image = batches.pad_image(image, padding_size)
    np.testing.assert_equal(padded_image, expected)


@pytest.mark.parametrize(
    "image, internal_size, padding_size, possible_outputs",
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
        image: np.ndarray, internal_size: int, padding_size: int,
        possible_outputs: List[np.ndarray]
) -> None:
    """Test `sample_square_part` function."""
    output = batches.sample_square_part(image, internal_size, padding_size)
    print(output)
    assert any([np.array_equal(output, arr) for arr in possible_outputs])


@pytest.mark.parametrize(
    "part, padding_size, expected_center, expected_frame",
    [
        (
            np.array([
                [[0, 1, 0],
                 [1, 0.5, 1],
                 [0, 1, 0]],
                [[1, 0, 1],
                 [0, 1, 0],
                 [1, 0, 0]]
            ]),
            1,
            np.array([
                [[0.5]],
                [[1]]
            ]),
            np.array([
                [[0, 1, 0],
                 [1, 0, 1],
                 [0, 1, 0]],
                [[1, 0, 1],
                 [0, 0, 0],
                 [1, 0, 0]]
            ])
        )
    ]
)
def test_split_into_center_and_frame(
        part: np.ndarray, padding_size: int,
        expected_center: np.ndarray, expected_frame: np.ndarray
) -> None:
    """Test `split_into_center_and_frame` function."""
    center, frame = batches.split_into_center_and_frame(part, padding_size)
    np.testing.assert_equal(center, expected_center)
    np.testing.assert_equal(frame, expected_frame)


@pytest.mark.parametrize(
    "part, padding_size",
    [
        (
            np.array([
                [[0, 1, 0],
                 [1, 0.5, 1],
                 [0, 1, 0]],
                [[1, 0, 1],
                 [0, 1, 0],
                 [1, 0, 0]]
            ]),
            1
        )
    ]
)
def test_generate_noisy_negative_example(
        part: np.ndarray, padding_size: int
) -> None:
    """Test `generate_noisy_negative_example` function."""
    fake_part = batches.generate_noisy_negative_example(part, padding_size)
    assert fake_part.min() >= 0
    assert fake_part.max() <= 1
    diff = part - fake_part
    assert diff[:, :padding_size, :].max() == 0
    assert diff[:, -padding_size:, :].max() == 0
    assert diff[:, :, :padding_size].max() == 0
    assert diff[:, :, -padding_size:].max() == 0


@pytest.mark.parametrize(
    "part, padding_size",
    [
        (
            np.array([
                [[0, 1, 0],
                 [1, 0.5, 1],
                 [0, 1, 0]],
                [[1, 0, 1],
                 [0, 1, 0],
                 [1, 0, 0]]
            ]),
            1
        )
    ]
)
def test_generate_blurry_negative_example(
        part: np.ndarray, padding_size: int
) -> None:
    """Test `generate_blurry_negative_example` function."""
    fake_part = batches.generate_blurry_negative_example(part, padding_size)
    assert fake_part.min() >= 0
    assert fake_part.max() <= 1
    diff = part - fake_part
    assert diff[:, :padding_size, :].max() == 0
    assert diff[:, -padding_size:, :].max() == 0
    assert diff[:, :, :padding_size].max() == 0
    assert diff[:, :, -padding_size:].max() == 0


@pytest.mark.parametrize(
    "part, another_part, padding_size, expected",
    [
        (
            np.array([
                [[0, 1, 0],
                 [1, 0.5, 1],
                 [0, 1, 0]],
                [[1, 0, 1],
                 [0, 1, 0],
                 [1, 0, 0]]
            ]),
            np.array([
                [[1, 0, 0],
                 [1, 0.8, 1],
                 [0, 1, 0]],
                [[0, 1, 1],
                 [0, 0.3, 1],
                 [1, 0, 0]]
            ]),
            1,
            np.array([
                [[0, 1, 0],
                 [1, 0.8, 1],
                 [0, 1, 0]],
                [[1, 0, 1],
                 [0, 0.3, 0],
                 [1, 0, 0]]
            ]),
        )
    ]
)
def test_generate_mismatching_negative_example(
        part: np.ndarray, another_part: np.ndarray, expected: np.ndarray,
        padding_size: int
) -> None:
    """Test `generate_mismatching_negative_example` function."""
    fake_part = batches.generate_mismatching_negative_example(
        part, another_part, padding_size
    )
    np.testing.assert_equal(fake_part, expected)
