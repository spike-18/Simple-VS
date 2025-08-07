import numpy as np
import matplotlib.colors as mcolors

from skimage.color import rgb2gray, rgba2rgb

from .util import plot_map, sobel_grad, plot_interpretation


def process_image(input_img: np.ndarray) -> np.ndarray:
    image = rgba2rgb(input_img) if input_img.shape[2] == 4 else input_img

    image = image[::4, ::4]

    height, width = image.shape[0], image.shape[1]

    points = np.zeros((height * width, 3), dtype=np.float32)
    points[:, 0] = np.arange(0, height * width) % width
    points[:, 2] = np.arange(0, height * width) // width
    # points[:, 1] = image.reshape((height * width, 3))[:, 0] * 100

    vert_edges, horiz_edges = get_edges(image, threshold=0.1)
    ground_map = get_ground(image)
    # ground_map = np.logical_or(ground_map, vert_edges)
    # ground_map = np.logical_or(ground_map, horiz_edges)

    # plot_map(image, ~ground_map, map_name="Figures", save=False)

    # Set y = 0 to all background pixels

    ground_map = ~ground_map.reshape(height*width,).astype(bool)
    points[ground_map, 1] = 0








    # plot_interpretation(image, points)



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


def get_ground(image: np.ndarray, sat_tres=0.5, val_tres=0.4) -> np.ndarray:
    image = rgba2rgb(image) if image.shape[2] == 4 else image

    hsv_image = mcolors.rgb_to_hsv(image)

    ground_map = np.logical_and(hsv_image[:,:,1] < sat_tres, hsv_image[:,:,2] > val_tres)

    plot_map(image, ~ground_map, map_name="Figures", save=False)

    return ground_map
