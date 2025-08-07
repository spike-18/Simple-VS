import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from matplotlib import cm
from matplotlib.colors import LightSource


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

    output[1 : x - 1, 1 : y - 1, 0] = (
        -image[: x - 2, 2:y]
        + image[2:x, 2:y]
        - 2 * image[: x - 2, 1 : y - 1]
        + 2 * image[2:x, 1 : y - 1]
        - image[: x - 2, 0 : y - 2]
        + image[2:x, 0 : y - 2]
    )

    output[1 : x - 1, 1 : y - 1, 1] = (
        -image[: x - 2, 2:y]
        - image[2:x, 2:y]
        - 2 * image[1 : x - 1, 2:y]
        + 2 * image[1 : x - 1, 0 : y - 2]
        + image[: x - 2, 0 : y - 2]
        + image[2:x, 0 : y - 2]
    )

    output[0, :, 0] = image[1, :] - image[0, :]
    output[x - 1, :, 0] = image[x - 1, :] - image[x - 2, :]
    output[: x - 1, 0, 0] = image[1:, 0] - image[: x - 1, 0]
    output[: x - 1, y - 1, 0] = image[1:, y - 1] - image[: x - 1, y - 1]

    output[:, 0, 1] = image[:, 1] - image[:, 0]
    output[:, y - 1, 1] = image[:, y - 1] - image[:, y - 2]
    output[0, : y - 1, 1] = image[0, 1:] - image[0, : y - 1]
    output[x - 1, : y - 1, 1] = image[x - 1, 1:] - image[x - 1, : y - 1]

    return output


def plot_map(image: np.ndarray, image_map: np.ndarray, save: bool, map_name="Map") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[1].imshow(image_map, cmap="gray")
    axes[1].set_title(map_name)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    if save:
        plt.savefig("example_images/" + map_name)
    plt.show()


def plot_interpretation(image: np.ndarray, points: np.ndarray) -> None:

    H, W = image.shape[0], image.shape[1]
    points[:, 1] = -points[:, 1]

    plotter = pv.Plotter()

    mesh = pv.StructuredGrid()
    mesh.points = points
    mesh.dimensions = [W, H, 1]

    plotter.add_mesh(mesh, show_edges=False)

    plotter.show_grid()
    plotter.camera_position = "xz"
    plotter.show()
