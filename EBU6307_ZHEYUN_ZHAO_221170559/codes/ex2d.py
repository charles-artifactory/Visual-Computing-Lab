import numpy as np
import cv2
import os

INPUT_DIR = "../inputs/"
OUTPUT_DIR = "../results/"


def gaussian_kernel(size, sigma):
    """
    Create a Gaussian kernel.

    Args:
        size (int): Kernel size.
        sigma (float): Standard deviation.

    Returns:
        np.ndarray: Gaussian kernel.
    """
    kernel = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return kernel / np.sum(kernel)


def convolve(image, kernel):
    """
    Perform convolution on an image.

    Args:
        image (np.ndarray): Input image.
        kernel (np.ndarray): Convolution kernel.

    Returns:
        np.ndarray: Convolved image.
    """
    height, width = image.shape
    k_height, k_width = kernel.shape
    pad_h, pad_w = k_height // 2, k_width // 2
    padded = np.zeros((height + 2 * pad_h, width + 2 * pad_w))
    padded[pad_h:pad_h + height, pad_w:pad_w + width] = image
    output = np.zeros_like(image, dtype=float)
    for i in range(height):
        for j in range(width):
            output[i, j] = np.sum(padded[i:i + k_height, j:j + k_width] * kernel)
    return output


def gaussian_blur(image, sigma=1.4):
    """
    Apply Gaussian blur to an image.

    Args:
        image (np.ndarray): Input image.
        sigma (float): Standard deviation for Gaussian kernel.

    Returns:
        np.ndarray: Blurred image.
    """
    size = int(6 * sigma + 1)
    if size % 2 == 0:
        size += 1
    kernel = gaussian_kernel(size, sigma)
    return convolve(image, kernel)


def sobel_filters(image):
    """
    Compute gradient magnitude and direction using Sobel operator.

    Args:
        image (np.ndarray): Input image.

    Returns:
        tuple: (magnitude, angle) of gradients.
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    G_x = convolve(image, sobel_x)
    G_y = convolve(image, sobel_y)
    magnitude = np.sqrt(G_x**2 + G_y**2)
    angle = np.arctan2(G_y, G_x)
    return magnitude, angle


def non_max_suppression(magnitude, angle):
    """
    Apply non-maximum suppression to gradient magnitude.

    Args:
        magnitude (np.ndarray): Gradient magnitude.
        angle (np.ndarray): Gradient direction.

    Returns:
        np.ndarray: Suppressed image.
    """
    height, width = magnitude.shape
    result = np.zeros_like(magnitude)
    angle = np.degrees(angle) % 180
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            q = 255
            r = 255
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]
            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                result[i, j] = magnitude[i, j]
    return result


def double_threshold(image, low_threshold=0.5, high_threshold=1.0):
    """
    Apply double thresholding to an image.

    Args:
        image (np.ndarray): Input image.
        low_threshold (float): Low threshold.
        high_threshold (float): High threshold.

    Returns:
        np.ndarray: Thresholded image with strong and weak edges.
    """
    result = np.zeros_like(image)
    strong_i, strong_j = np.where(image >= high_threshold)
    weak_i, weak_j = np.where((image < high_threshold) & (image >= low_threshold))
    result[strong_i, strong_j] = 1
    result[weak_i, weak_j] = 0.5
    return result


def hysteresis(image):
    """
    Perform edge tracking by hysteresis.

    Args:
        image (np.ndarray): Double thresholded image.

    Returns:
        np.ndarray: Final edge map.
    """
    height, width = image.shape
    result = np.copy(image)
    dx = [-1, -1, -1, 0, 0, 1, 1, 1]
    dy = [-1, 0, 1, -1, 1, -1, 0, 1]
    strong_edges = (result == 1)
    weak_edges = (result == 0.5)
    while True:
        prev_result = np.copy(result)
        for i in range(1, height-1):
            for j in range(1, width-1):
                if weak_edges[i, j]:
                    for k in range(8):
                        ni, nj = i + dx[k], j + dy[k]
                        if 0 <= ni < height and 0 <= nj < width and result[ni, nj] == 1:
                            result[i, j] = 1
                            break
        if np.array_equal(prev_result, result):
            break
    result[result != 1] = 0
    return result


def canny_edge_detector(image, sigma=1.4):
    """
    Perform Canny edge detection.

    Args:
        image (np.ndarray): Input grayscale image.
        sigma (float): Gaussian blur parameter.

    Returns:
        tuple: (blurred, magnitude, suppressed, thresholded, edges)
    """
    blurred = gaussian_blur(image, sigma)
    magnitude, angle = sobel_filters(blurred)
    suppressed = non_max_suppression(magnitude, angle)
    if suppressed.max() > 0:
        normalized = suppressed * 10 / suppressed.max()
    else:
        normalized = suppressed
    thresholded = double_threshold(normalized, 0.5, 1.0)
    edges = hysteresis(thresholded)
    return blurred, magnitude, suppressed, thresholded, edges


def main():
    """
    Main function to run Canny edge detection and save results.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    image_path = os.path.join(INPUT_DIR, "building.jpg")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Cannot read image: {image_path}")
        return
    blurred, magnitude, suppressed, thresholded, edges = canny_edge_detector(image)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2d_canny1.jpg"), blurred)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2d_canny2.jpg"), magnitude)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2d_canny3.jpg"), suppressed)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2d_canny4.jpg"), thresholded * 255)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2d_canny5.jpg"), edges * 255)
    opencv_edges = cv2.Canny(image, 100, 200)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2d_canny6.jpg"), opencv_edges)
    print("Canny edge detection completed. Results saved to output directory.")


if __name__ == "__main__":
    print('ex2d...')
    main()
