import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray, rgba2rgb

from .util import get_gradient_map


def process_image(input_img: np.ndarray) -> np.ndarray:
    gray_img = rgb2gray(rgba2rgb(input_img))

    grad_map = get_gradient_map(gray_img)
    grad_norm_map = np.linalg.norm(grad_map, axis=2)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    axes[0].imshow(input_img)
    axes[0].set_title("Input Image")
    axes[1].imshow(grad_norm_map)
    axes[1].set_title("Gradient Norm Map")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
