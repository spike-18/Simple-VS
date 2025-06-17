import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgba2rgb, rgb2gray


def process_image(input_img: np.ndarray) -> np.ndarray:

    input_img = rgba2rgb(input_img)

    grad_map = get_gradient_map(input_img)
    grad_norm_map = np.linalg.norm(grad_map, axis=2)


    fig, axes = plt.subplots(2, 2)

    axes[0, 0].imshow(input_img)
    axes[0, 0].set_title("Initial image")

    axes[0, 1].imshow(grad_norm_map[:,:,0])
    axes[0, 1].set_title("Red channel")

    axes[1, 0].imshow(grad_norm_map[:,:,1])
    axes[1, 0].set_title("Green channel")

    axes[1, 1].imshow(grad_norm_map[:,:,2])
    axes[1, 1].set_title("Blue channel")

    for ax in axes:
        ax[0].axis("off")
        ax[1].axis("off")

    plt.tight_layout()
    plt.show()



def get_gradient_map(image: np.ndarray) -> np.ndarray:

    x, y = image.shape[0], image.shape[1]
    output = np.zeros((x, y, 2, 3), dtype=np.float32)

    output[1:,:, 0, :] = image[1:,:, :] - image[:x-1,:, :]
    output[:,1:, 1, :] = image[:,1:, :] - image[:,:y-1, :]
    output[0, :, 0, :] = image[1, :, :] - image[0,:, :]
    output[:, 0, 1, :] = image[:, 1, :] - image[:,0, :]

    return output
