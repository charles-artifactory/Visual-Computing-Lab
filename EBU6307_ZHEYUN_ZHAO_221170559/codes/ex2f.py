import numpy as np
import cv2
import os
import math

INPUT_DIR = "../inputs/"
OUTPUT_DIR = "../results/"


def sobel_edge_detection(image):
    """
    Detect edges using Sobel operators

    Args:
        image: Input grayscale image

    Returns:
        Edge map with gradient magnitude
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    height, width = image.shape

    padded_img = np.zeros((height + 2, width + 2))
    padded_img[1:height+1, 1:width+1] = image

    gradient_x = np.zeros((height, width))
    gradient_y = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            roi = padded_img[i:i+3, j:j+3]
            gradient_x[i, j] = np.sum(roi * sobel_x)
            gradient_y[i, j] = np.sum(roi * sobel_y)

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    if gradient_magnitude.max() > 0:
        gradient_magnitude = gradient_magnitude * 255.0 / gradient_magnitude.max()

    binary_edge_map = (gradient_magnitude > 128).astype(np.uint8) * 255

    return binary_edge_map


def hough_transform(edge_map):
    """
    Apply Hough Transform to detect lines in edge map

    Args:
        edge_map: Binary edge map from edge detection

    Returns:
        Hough accumulator array in (rho, theta) space
    """
    height, width = edge_map.shape

    theta_range = np.deg2rad(np.arange(0, 180))

    max_rho = int(np.sqrt(height**2 + width**2))

    rho_range = np.arange(-max_rho, max_rho)

    accumulator = np.zeros((len(rho_range), len(theta_range)))

    y_indices, x_indices = np.where(edge_map > 0)

    for i in range(len(y_indices)):
        y = y_indices[i]
        x = x_indices[i]

        for theta_idx, theta in enumerate(theta_range):
            rho = int(x * np.cos(theta) + y * np.sin(theta))

            rho_idx = rho + max_rho

            if 0 <= rho_idx < len(rho_range):
                accumulator[rho_idx, theta_idx] += 1

    return accumulator, rho_range, theta_range


def detect_lines(accumulator, rho_range, theta_range, threshold_ratio=0.5):
    """
    Detect lines from Hough accumulator using thresholding

    Args:
        accumulator: Hough accumulator array
        rho_range: Range of rho values
        theta_range: Range of theta values
        threshold_ratio: Threshold ratio relative to max value

    Returns:
        List of detected lines as (rho, theta) pairs
    """
    threshold = threshold_ratio * accumulator.max()

    rho_indices, theta_indices = np.where(accumulator >= threshold)

    lines = []
    for i in range(len(rho_indices)):
        rho = rho_range[rho_indices[i]]
        theta = theta_range[theta_indices[i]]
        lines.append((rho, theta))

    return lines


def draw_lines(image, lines, rho_offset):
    """
    Draw detected lines on the image

    Args:
        image: Input image
        lines: List of detected lines as (rho, theta) pairs
        rho_offset: Offset for rho values

    Returns:
        Image with drawn lines
    """
    result = image.copy()
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    # For each detected line
    height, width = image.shape[:2]
    for rho, theta in lines:
        rho = rho - rho_offset

        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return result


def visualize_hough_space(accumulator):
    """
    Visualize Hough transform space

    Args:
        accumulator: Hough accumulator array

    Returns:
        Visualization of Hough space
    """
    hough_image = accumulator.copy()
    if hough_image.max() > 0:
        hough_image = (hough_image * 255.0 / hough_image.max()).astype(np.uint8)

    hough_image = cv2.resize(hough_image, (500, 500))

    return hough_image


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Read the input image
    image_path = os.path.join(INPUT_DIR, "HoughTransformLines.jpg")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return

    print(f"Processing image: {image_path}")
    print(f"Image shape: {image.shape}")

    # Step 1: Create edge map using Sobel operators
    print("Step 1: Creating edge map using Sobel operators...")
    edge_map = sobel_edge_detection(image)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2f_edgemap.jpg"), edge_map)
    print(f"Saved edge map to {os.path.join(OUTPUT_DIR, 'ex2f_edgemap.jpg')}")

    # Step 2: Apply custom Hough Transform
    print("Step 2: Applying custom Hough Transform...")
    accumulator, rho_range, theta_range = hough_transform(edge_map)
    hough_image = visualize_hough_space(accumulator)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2f_houghtransform1.jpg"), hough_image)
    print(f"Saved Hough Transform visualization to {os.path.join(OUTPUT_DIR, 'ex2f_houghtransform1.jpg')}")

    # Step 3: Detect lines using threshold
    print("Step 3: Detecting lines from Hough Transform...")
    lines = detect_lines(accumulator, rho_range, theta_range, threshold_ratio=0.5)
    print(f"Detected {len(lines)} lines with custom Hough Transform")

    # Step 4: Draw detected lines
    result_image = draw_lines(image, lines, rho_offset=len(rho_range) // 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2f_detectedline1.jpg"), result_image)
    print(f"Saved detected lines image to {os.path.join(OUTPUT_DIR, 'ex2f_detectedline1.jpg')}")

    # Step 5: Repeat using OpenCV functions
    print("Step 4: Applying OpenCV's Hough Transform...")

    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    lines_cv = cv2.HoughLines(edges, 1, np.pi/180, 150)

    accumulator_cv = np.zeros((edge_map.shape[0] * 2, 180))
    for rho, theta in lines_cv[:, 0, :]:
        theta_idx = int(theta * 180 / np.pi)
        rho_idx = int(rho) + edge_map.shape[0]
        if 0 <= theta_idx < 180 and 0 <= rho_idx < accumulator_cv.shape[0]:
            accumulator_cv[rho_idx, theta_idx] = 255

    hough_image_cv = visualize_hough_space(accumulator_cv)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2f_houghtransform2.jpg"), hough_image_cv)
    print(f"Saved OpenCV Hough Transform visualization to {os.path.join(OUTPUT_DIR, 'ex2f_houghtransform2.jpg')}")

    result_image_cv = image.copy()
    if len(result_image_cv.shape) == 2:
        result_image_cv = cv2.cvtColor(result_image_cv, cv2.COLOR_GRAY2BGR)

    if lines_cv is not None:
        for line in lines_cv:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(result_image_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2f_detectedline2.jpg"), result_image_cv)
    print(f"Saved OpenCV detected lines image to {os.path.join(OUTPUT_DIR, 'ex2f_detectedline2.jpg')}")

    print("Hough Transform processing completed successfully!")


if __name__ == "__main__":
    main()
