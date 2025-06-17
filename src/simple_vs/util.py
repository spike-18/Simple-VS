import numpy as np


def get_gradient_map(image: np.ndarray) -> np.ndarray:
    x, y = image.shape[0], image.shape[1]
    output = np.zeros((x, y, 2), dtype=np.float32)

    output[1:, :, 0] = image[1:, :] - image[: x - 1, :]
    output[:, 1:, 1] = image[:, 1:] - image[:, : y - 1]
    output[0, :, 0] = image[1, :] - image[0, :]
    output[:, 0, 1] = image[:, 1] - image[:, 0]

    return output
