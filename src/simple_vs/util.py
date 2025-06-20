import matplotlib.pyplot as plt
import numpy as np
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


def plot_interpretation(image: np.ndarray, coords: tuple[np.array]) -> None:

    x_img = coords[0]
    y_img = coords[1]
    z_img = coords[2]

    H, W = image.shape[0], image.shape[1]

    X_3d = np.linspace(np.min(x_img), np.max(x_img), W)
    Y_3d = np.linspace(np.min(z_img), np.max(z_img), H)

    X, Y = np.meshgrid(X_3d, Y_3d)
    Z = np.full((Y_3d.size, X_3d.size), np.nan)

    



    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, Z, rstride=1)
    ax.scatter(x_img, z_img, y_img, color="red")

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")

    plt.show()
