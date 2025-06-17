import numpy as np


def get_gradient_map(image: np.ndarray) -> np.ndarray:
    x, y = image.shape[0], image.shape[1]
    output = np.zeros((x, y, 2), dtype=np.float32)

    output[1:, :, 0] = image[1:, :] - image[: x - 1, :]
    output[:, 1:, 1] = image[:, 1:] - image[:, : y - 1]
    output[0, :, 0] = image[1, :] - image[0, :]
    output[:, 0, 1] = image[:, 1] - image[:, 0]

    return output


def sobel_grad(image: np.ndarray) -> np.ndarray:
    x, y = image.shape[0], image.shape[1]
    output = np.zeros((x, y, 2), dtype=np.float32)

    output[1:x-1, 1:y-1, 0] = (-image[:x-2, 2:y] + image[2:x,2:y] - 2*image[:x-2, 1:y-1] \
                                    + 2*image[2:x, 1:y-1] - image[:x-2, 0:y-2] + image[2:x,0:y-2])

    output[1:x-1, 1:y-1, 1] = (-image[:x-2, 2:y] - image[2:x,2:y] - 2*image[1:x-1, 2:y] \
                                    + 2*image[1:x-1, 0:y-2] + image[:x-2, 0:y-2] + image[2:x,0:y-2])

    output[0, :, 0] = image[1, :] - image[0, :]
    output[x-1, :, 0] = image[x-1, :] - image[x-2, :]
    output[:x-1, 0, 0] = image[1:, 0] - image[:x-1, 0]
    output[:x-1, y-1, 0] = image[1:, y-1] - image[:x-1, y-1]

    output[:, 0, 1] = image[:, 1] - image[:, 0]
    output[:, y-1, 1] = image[:, y-1] - image[:, y-2]
    output[0, :y-1, 1] = image[0, 1:] - image[0, :y-1]
    output[x-1, :y-1, 1] = image[x-1, 1:] - image[x-1, :y-1]

    return output
