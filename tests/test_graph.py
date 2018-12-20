"""
Test `graph.py` module.

Author: Nikolay Lysenko
"""


from typing import List, Tuple

import pytest
import tensorflow as tf
import numpy as np

from gpn.graph import sample_multiple_fragments


@pytest.mark.parametrize(
    "images, corners, fragment_size, frame_size, n_channels, expected",
    [
        (
            # `images`
            np.array([
                [
                    [[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [1, 0, 1, 0],
                     [0, 1, 0, 1]],
                    [[1, 1, 1, 1],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]],
                    [[0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [1, 1, 1, 1]]
                ],
                [
                    [[1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]],
                    [[0, 0, 0, 0],
                     [0, 1, 1, 0],
                     [0, 1, 1, 0],
                     [0, 0, 0, 0]],
                    [[0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 1, 1],
                     [0, 0, 1, 1]]
                ]
            ]).swapaxes(1, 3),
            # `corners`
            [(1, 1), (0, 2)],
            # `fragment_size`
            4,
            # `frame_size`
            1,
            # `n_channels`
            3,
            # `expected`
            np.array([
                [
                    [[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [1, 0, 1, 0],
                     [0, 1, 0, 1]],
                    [[1, 1, 1, 1],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]],
                    [[0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [1, 1, 1, 1]]
                ],
                [
                    [[1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]],
                    [[0, 0, 0, 0],
                     [0, 1, 1, 0],
                     [0, 1, 1, 0],
                     [0, 0, 0, 0]],
                    [[0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 1, 1],
                     [0, 0, 1, 1]]
                ],
                [
                    [[0, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 0]],
                    [[0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]],
                    [[0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 1, 1, 1],
                     [0, 0, 0, 0]]
                ],
                [
                    [[0, 1, 1, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]],
                    [[0, 0, 1, 1],
                     [0, 0, 1, 1],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]],
                    [[0, 0, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 0, 1],
                     [0, 0, 0, 0]]
                ],
            ]).swapaxes(1, 3)
        )
    ]
)
def test_sample_multiple_fragments(
        images: np.ndarray, corners: List[Tuple[int, int]],
        fragment_size: int, frame_size: int, n_channels: int,
        expected: np.ndarray
) -> None:
    """Test `sample_multiple_fragments` function."""
    graph = tf.Graph()
    with graph.as_default():
        tensor_images = tf.placeholder(tf.float32, images.shape)
        tensor_corners = [
            tf.placeholder(tf.int32, (2,), name=f'corner_{i}')
            for i, _ in enumerate(corners)
        ]
        tensor_fragments = sample_multiple_fragments(
            tensor_images, tensor_corners,
            fragment_size, frame_size, n_channels
        )
    with tf.Session(graph=graph) as sess:
        feed_dict = {
            tensor_images: images,
            **{k: v for k, v in zip(tensor_corners, corners)}
        }
        fragments = tensor_fragments.eval(feed_dict, sess)
    np.testing.assert_array_equal(fragments, expected)
