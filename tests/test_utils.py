"""
Test `utils.py` module.

Author: Nikolay Lysenko
"""


from typing import List

import pytest
import numpy as np

from gpn.utils import shuffle_multiple_arrays


@pytest.mark.parametrize(
    "arrays",
    [
        ([np.array([1, 2, 3, 4, 5]), np.array([0, 1, 1, 0, 0])]),
        ([np.array([[1, 2, 3], [4, 5, 6]]), np.array([0, 1])]),
    ]
)
def test_shuffle_multiple_arrays(arrays: List[np.ndarray]) -> None:
    """Test `shuffle_multiple_arrays` function."""
    two_dim_arrays = [
        arr.reshape([-1, 1]) if len(arr.shape) < 2 else arr
        for arr in arrays
    ]
    expected = np.hstack(two_dim_arrays)
    shuffle_multiple_arrays(*arrays)
    two_dim_arrays = [
        arr.reshape([-1, 1]) if len(arr.shape) < 2 else arr
        for arr in arrays
    ]
    result = np.hstack(two_dim_arrays)
    result = result[result[:, 0].argsort()]
    np.testing.assert_array_equal(result, expected)
