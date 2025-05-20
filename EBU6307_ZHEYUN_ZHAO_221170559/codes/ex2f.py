import numpy as np
import cv2
import os
from math import sin, cos, radians, pi

INPUT_DIR = "../inputs/"
OUTPUT_DIR = "../results/"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def sobel_operator(img):
    """
    Generate edge map using Sobel operator.

    Args:
        img (np.ndarray): Grayscale input image.

    Returns:
        np.ndarray: Edge map as uint8 image.
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    padded_img = np.pad(img, ((1, 1), (1, 1)), mode='constant')
    height, width = img.shape
    edge_map = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            region = padded_img[i:i+3, j:j+3]
            gx = np.sum(region * sobel_x)
            gy = np.sum(region * sobel_y)
            edge_map[i, j] = np.sqrt(gx**2 + gy**2)
    if edge_map.max() > 0:
        edge_map = edge_map / edge_map.max() * 255
    return edge_map.astype(np.uint8)


def custom_hough_transform(edge_map, threshold_ratio=0.5):
    """
    Custom implementation of Hough transform for line detection.

    Args:
        edge_map (np.ndarray): Edge map image.
        threshold_ratio (float): Ratio for thresholding accumulator.

    Returns:
        tuple: (Hough accumulator array, list of detected lines as (rho, theta))
    """
    height, width = edge_map.shape
    diagonal = int(np.sqrt(height**2 + width**2))
    rho_range = np.linspace(-diagonal, diagonal, 2*diagonal)
    theta_range = np.linspace(0, 180, 180)
    hough_space = np.zeros((len(rho_range), len(theta_range)))
    y_idxs, x_idxs = np.nonzero(edge_map)
    for i in range(len(y_idxs)):
        y = y_idxs[i]
        x = x_idxs[i]
        for theta_idx, theta in enumerate(theta_range):
            rho = x * cos(radians(theta)) + y * sin(radians(theta))
            rho_idx = int(rho + diagonal)
            if 0 <= rho_idx < len(rho_range):
                hough_space[rho_idx, theta_idx] += 1
    hough_norm = hough_space / hough_space.max() if hough_space.max() > 0 else hough_space
    threshold = threshold_ratio * hough_space.max()
    rho_idxs, theta_idxs = np.where(hough_space > threshold)
    lines = []
    for i in range(len(rho_idxs)):
        rho = rho_range[rho_idxs[i]]
        theta = theta_range[theta_idxs[i]]
        lines.append((rho, theta))
    return hough_space, lines


def draw_lines(img, lines, color=(0, 0, 255)):
    """
    Draw detected lines on the image.

    Args:
        img (np.ndarray): Input image.
        lines (list): List of (rho, theta) tuples.
        color (tuple): Line color in BGR.

    Returns:
        np.ndarray: Image with lines drawn.
    """
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
    height, width = img.shape[:2]
    for rho, theta in lines:
        a = cos(radians(theta))
        b = sin(radians(theta))
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(result, (x1, y1), (x2, y2), color, 2)
    return result


def visualize_hough_space(hough_space, output_path):
    """
    Visualize Hough accumulator space and save as image.

    Args:
        hough_space (np.ndarray): Hough accumulator array.
        output_path (str): Output file path.
    """
    normalized = ((hough_space / hough_space.max()) * 255).astype(np.uint8)
    resized_hough = cv2.resize(normalized, (800, 600))
    cv2.imwrite(output_path, resized_hough)


def main():
    """
    Main function to perform edge detection, Hough transform, and visualization.
    """
    input_path = os.path.join(INPUT_DIR, "HoughTransformLines.jpg")
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return
    edge_map = sobel_operator(img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2f_edgemap.jpg"), edge_map)
    hough_space, lines = custom_hough_transform(edge_map, threshold_ratio=0.5)
    visualize_hough_space(hough_space, os.path.join(OUTPUT_DIR, "ex2f_houghtransform1.jpg"))
    result_custom = draw_lines(img, lines)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2f_detectedline1.jpg"), result_custom)
    edges = cv2.Canny(img, 50, 150)
    lines_cv = cv2.HoughLines(edges, 1, np.pi/180,
                              threshold=int(0.5*np.max(cv2.HoughLines(edges, 1, np.pi/180, 50))))
    cv_hough_space = np.zeros((2*int(np.sqrt(img.shape[0]**2 + img.shape[1]**2)), 180))
    diagonal = int(np.sqrt(img.shape[0]**2 + img.shape[1]**2))
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if edges[y, x] > 0:
                for theta_idx in range(180):
                    theta = theta_idx * np.pi / 180
                    rho = x * np.cos(theta) + y * np.sin(theta)
                    rho_idx = int(rho + diagonal)
                    if 0 <= rho_idx < 2*diagonal:
                        cv_hough_space[rho_idx, theta_idx] += 1
    visualize_hough_space(cv_hough_space, os.path.join(OUTPUT_DIR, "ex2f_houghtransform2.jpg"))
    result_cv = img.copy()
    result_cv = cv2.cvtColor(result_cv, cv2.COLOR_GRAY2BGR)
    if lines_cv is not None:
        for rho, theta in lines_cv[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(result_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2f_detectedline2.jpg"), result_cv)


if __name__ == "__main__":
    print('ex2f...')
    main()
