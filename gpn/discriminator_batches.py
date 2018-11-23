"""
Prepare batches for discriminator.

Author: Nikolay Lysenko
"""


from typing import Tuple, Generator

import numpy as np


def pad_image(image: np.ndarray, padding_size: int) -> np.ndarray:
    """
    Create padding with zeros for all spatial dimensions.

    :param image:
        array of shape (channel_dim, x_dim, y_dim)
    :param padding_size:
        width in pixels of outer padding to be created
    :return:
        padded array
    """
    n_channels = image.shape[0]
    initial_width = image.shape[2]
    vertical_padding = np.zeros((n_channels, padding_size, initial_width))
    arrays = (vertical_padding, image, vertical_padding)
    image = np.concatenate(arrays, axis=1)
    new_height = image.shape[1]
    horizontal_padding = np.zeros((n_channels, new_height, padding_size))
    arrays = (horizontal_padding, image, horizontal_padding)
    image = np.concatenate(arrays, axis=2)
    return image


def sample_square_part(
        image: np.ndarray, internal_size: int, frame_size: int
) -> np.ndarray:
    """
    Sample a square fragment of an image.

    Such fragment can be split into two parts:
    * its center of size `internal_size * internal_size`,
    * a frame around the center of width equal to `frame_size`.
    It is assumed that the center must be entirely within
    the original image, but frame may go beyond it.

    :param image:
        array of shape (channel_dim, x_dim, y_dim)
    :param internal_size:
        size of square that must be entirely within the original image
    :param frame_size:
        size of a frame that can be both inside or outside of the image
    :return:
        square image of size `internal_size + 2 * frame_size`
    """
    x_options = list(range(image.shape[1] - internal_size + 1))
    x_corner = np.random.choice(x_options)
    y_options = list(range(image.shape[2] - internal_size + 1))
    y_corner = np.random.choice(y_options)
    x_min = x_corner
    x_max = x_corner + internal_size + 2 * frame_size
    y_min = y_corner
    y_max = y_corner + internal_size + 2 * frame_size
    padded_image = pad_image(image, frame_size)
    part = padded_image[:, x_min:x_max, y_min:y_max]
    return part


def split_into_center_and_frame(
        part: np.ndarray, frame_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split image part into its center and an internal frame around the center.

    :param part:
        sampled fragment of an image of shape (channel_dim, x_dim, y_dim)
    :param frame_size:
        size of a frame to be separated
    :return:
        a tuple of a central part without padding
        and a frame with zeros at center
    """
    center = part[:, frame_size:-frame_size, frame_size:-frame_size]
    padded_center = pad_image(center, frame_size)
    frame = part - padded_center
    return center, frame


def generate_noisy_negative_example(
        part: np.ndarray, frame_size: int
) -> np.ndarray:
    """
    Generate noisy negative example from a real positive example.

    :param part:
        sampled fragment of a real image of shape (channel_dim, x_dim, y_dim)
    :param frame_size:
        size of an internal frame around center of `part`
    :return:
        fake fragment of an image where center is a noise
        and frame around it is the same as in the original `part`
    """
    center, frame = split_into_center_and_frame(part, frame_size)
    noisy_center = np.random.uniform(size=center.shape)
    noisy_center = pad_image(noisy_center, frame_size)
    fake_part = noisy_center + frame
    return fake_part


def generate_blurry_negative_example(
        part: np.ndarray, frame_size: int, noise_intensity: float = 0.2
) -> np.ndarray:
    """
    Generate blurry negative example from a real positive example.

    :param part:
        sampled fragment of a real image of shape (channel_dim, x_dim, y_dim)
    :param frame_size:
        size of an internal frame around center of `part`
    :param noise_intensity:
        positive number; the higher it is, the heavier blurring is
    :return:
        fake fragment of an image where center is blurred with noise
        and frame around it is the same as in the original `part`
    """
    center, frame = split_into_center_and_frame(part, frame_size)
    noisy_mask = np.random.uniform(
        size=center.shape, low=-noise_intensity, high=noise_intensity
    )
    noisy_mask = pad_image(noisy_mask, frame_size)
    fake_part = part + noisy_mask
    fake_part = fake_part.clip(min=0, max=1)
    return fake_part


def generate_mismatching_negative_example(
        part: np.ndarray, another_part: np.ndarray, frame_size: int
) -> np.ndarray:
    """
    Generate negative example with center taken from other part.

    :param part:
        sampled fragment of a real image of shape (channel_dim, x_dim, y_dim)
    :param another_part:
        sampled fragment of a real image of shape (channel_dim, x_dim, y_dim)
    :param frame_size:
        size of an internal frame around center of `part` or `another_part`
    :return:
        fake fragment where center and frame are mismatching
    """
    center, frame = split_into_center_and_frame(part, frame_size)
    another_center, another_frame = split_into_center_and_frame(
        another_part, frame_size
    )
    another_center = pad_image(another_center, frame_size)
    fake_part = another_center + frame
    return fake_part


def yield_discriminator_batches(
        images: np.ndarray, n_positives_per_batch: int,
        internal_size: int, frame_size: int
) -> Generator[Tuple[np.ndarray], None, None]:
    """
    Yield batches for discriminator training.

    :param images:
        array of real images of shape (n_images, n_channels, x_dim, y_dim)
    :param n_positives_per_batch:
        number of positive (real) examples per batch
    :param internal_size:
        size of square that must be entirely within the original image
    :param frame_size:
        size of a frame that can be outside of the original image
    :return:
        batches of shape (4 * n_positives_per_batch, n_channels, x_dim, y_dim)
        where for each original image there are:
        * its fragment,
        * the same fragment with noise at center,
        * the same fragment wih blurred center,
        * the same fragment with center taken from another image
    """
    size_of_incomplete_batch = images.shape[0] % n_positives_per_batch
    if size_of_incomplete_batch > 0:
        images = images[:-size_of_incomplete_batch, :, :, :]
    images = images.reshape(-1, n_positives_per_batch, *images.shape[1:])
    for i in range(images.shape[0]):
        positive_batch = images[i]
        features = []
        for j in range(positive_batch.shape[0]):
            image = positive_batch[j]

            real_part = sample_square_part(
                image, internal_size, frame_size
            )
            features.append(real_part.reshape(-1, *real_part.shape))

            noisy_part = generate_noisy_negative_example(
                real_part, frame_size
            )
            features.append(noisy_part.reshape(-1, *noisy_part.shape))

            blurry_part = generate_blurry_negative_example(
                real_part, frame_size
            )
            features.append(blurry_part.reshape(-1, *blurry_part.shape))

            another_batch = images[(i + 1) % images.shape[0]]
            another_image = another_batch[j]
            another_part = sample_square_part(
                another_image, internal_size, frame_size
            )
            mismatching_part = generate_mismatching_negative_example(
                real_part, another_part, frame_size
            )
            features.append(
                mismatching_part.reshape(-1, *mismatching_part.shape)
            )

        features = np.concatenate(features, axis=0)
        labels = np.repeat(np.array([1, 0, 0, 0]), n_positives_per_batch)
        yield features, labels
