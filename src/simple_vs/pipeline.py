import numpy as np

from skimage.color import rgb2gray, rgba2rgb

from .util import plot_map, sobel_grad, plot_interpretation


def process_image(input_img: np.ndarray) -> np.ndarray:
    image = rgba2rgb(input_img) if input_img.shape[2] == 4 else input_img


    # image = image[::10, ::10]

    height, width = image.shape[0], image.shape[1]

    points = np.zeros((height * width, 3), dtype=np.float32)
    points[:, 0] = np.arange(0, height * width) % width
    points[:, 1] = np.arange(0, height * width) // width
    points[:, 2] = image.reshape((height * width, 3))[:, 0] * 100

    vert_edges, horiz_edges = get_edges(image, threshold=0.1)
    fig_map = get_figures(image)

    plot_interpretation(image, points)
    
    


def get_edges(image: np.ndarray, threshold=0.15) -> tuple[np.ndarray, np.ndarray]:
    gray_img = rgb2gray(rgba2rgb(image)) if image.shape[2] == 4 else rgb2gray(image)

    grad_map = sobel_grad(gray_img)
    grad_norm_map = np.linalg.norm(grad_map, ord=2, axis=2)
    grad_norm_map = (grad_norm_map - np.min(grad_norm_map)) / (
        np.max(grad_norm_map) - np.min(grad_norm_map)
    )

    grad_angle_map = np.arctan2(grad_map[:, :, 1], grad_map[:, :, 0])

    edges_map = grad_norm_map > threshold

    angle_thresh = np.deg2rad(30)

    vertical_edges = edges_map & (
        (np.abs(grad_angle_map - np.pi / 2) < angle_thresh)
        | (np.abs(grad_angle_map + np.pi / 2) < angle_thresh)
    )

    horizontal_edges = ~vertical_edges & edges_map

    # color_edges = np.zeros((*vertical_edges.shape, 3), dtype=np.float32)
    # color_edges[vertical_edges] = [0, 1, 0]  # Green
    # color_edges[horizontal_edges] = [1, 0, 0]  # Red

    # plot_map(image, color_edges, map_name="Edges", save=False)

    return vertical_edges, horizontal_edges


def get_figures(image: np.ndarray, threshold=0.3) -> np.ndarray:
    image = rgba2rgb(image) if image.shape[2] == 4 else image

    norm_image = image.astype(np.float32) / 255.0

    max_c = np.max(norm_image, axis=2)
    min_c = np.min(norm_image, axis=2)
    delta = max_c - min_c

    sat_map = delta / max_c
    sat_map[np.isinf(sat_map)] = 0

    fig_map = sat_map > threshold

    # plot_map(image, fig_map, map_name="Figures", save=False)

    return fig_map
