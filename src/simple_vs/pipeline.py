import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray, rgba2rgb

from .util import get_gradient_map, sobel_grad


def process_image(input_img: np.ndarray) -> np.ndarray:
    edges_map = get_edges(input_img, threshold=0.1)



def get_edges(image: np.ndarray, threshold = 0.15) -> np.ndarray:
    gray_img = rgb2gray(rgba2rgb(image)) if image.shape[2] == 4 else rgb2gray(image)

    grad_map = sobel_grad(gray_img)
    grad_norm_map = np.linalg.norm(grad_map, ord=2, axis=2)
    grad_norm_map = (grad_norm_map - np.min(grad_norm_map)) / (np.max(grad_norm_map) -
                                                               np.min(grad_norm_map))

    grad_angle_map = np.arctan2(grad_map[:, :, 1], grad_map[:, :, 0])

    edges_map = (grad_norm_map > threshold)

    angle_thresh = np.deg2rad(30)

    vertical_edges = edges_map & (
        (np.abs(grad_angle_map - np.pi/2) < angle_thresh) |
        (np.abs(grad_angle_map + np.pi/2) < angle_thresh)
    )

    horizontal_edges = ~vertical_edges & edges_map

    color_edges = np.zeros((*vertical_edges.shape, 3), dtype=np.float32)
    color_edges[vertical_edges] = [0, 1, 0]   # Green
    color_edges[horizontal_edges] = [1, 0, 0] # Red


    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[1].imshow(color_edges)
    axes[1].set_title("Edges")


    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("example_images/edges_output.png")
    plt.show()
