import numpy as np
import cv2
import os
import math

INPUT_DIR = "../inputs/"
OUTPUT_DIR = "../results/"


def gaussian_kernel(size, sigma):
    """
    Create a 2D Gaussian kernel

    Args:
        size: Kernel size (must be odd)
        sigma: Standard deviation

    Returns:
        2D numpy array with the Gaussian kernel
    """
    if size % 2 == 0:
        size += 1

    center = size // 2

    kernel = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = (1 / (2 * math.pi * sigma**2)) * math.exp(-(x**2 + y**2) / (2 * sigma**2))

    return kernel / np.sum(kernel)


def convolution2d(image, kernel):
    """
    Apply 2D convolution on an image using the given kernel

    Args:
        image: Input grayscale image as numpy array
        kernel: 2D convolution kernel

    Returns:
        Convolved image
    """
    i_height, i_width = image.shape
    k_height, k_width = kernel.shape

    pad_h = k_height // 2
    pad_w = k_width // 2

    padded_img = np.zeros((i_height + 2 * pad_h, i_width + 2 * pad_w))
    padded_img[pad_h:pad_h + i_height, pad_w:pad_w + i_width] = image

    output = np.zeros_like(image)

    for i in range(i_height):
        for j in range(i_width):
            roi = padded_img[i:i + k_height, j:j + k_width]
            output[i, j] = np.sum(roi * kernel)

    return output


def gaussian_smoothing(image, sigma=1.4):
    """
    Apply Gaussian smoothing to an image

    Args:
        image: Input grayscale image
        sigma: Standard deviation of the Gaussian kernel

    Returns:
        Smoothed image
    """
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = gaussian_kernel(kernel_size, sigma)

    return convolution2d(image, kernel)


def sobel_operator(image):
    """
    Apply Sobel operator to calculate gradient magnitude and direction

    Args:
        image: Input grayscale image

    Returns:
        Tuple (gradient_magnitude, gradient_direction)
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    gradient_x = convolution2d(image, sobel_x)
    gradient_y = convolution2d(image, sobel_y)

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    return gradient_magnitude, gradient_direction


def non_max_suppression(gradient_magnitude, gradient_direction):
    """
    Apply non-maximum suppression to thin edges

    Args:
        gradient_magnitude: Gradient magnitude image
        gradient_direction: Gradient direction image in radians

    Returns:
        Suppressed edge image
    """
    degrees = np.degrees(gradient_direction) % 180

    height, width = gradient_magnitude.shape
    suppressed = np.zeros((height, width))

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            magnitude = gradient_magnitude[i, j]

            angle = degrees[i, j]

            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                neighbors = [gradient_magnitude[i, j-1], gradient_magnitude[i, j+1]]
            elif 22.5 <= angle < 67.5:
                neighbors = [gradient_magnitude[i-1, j+1], gradient_magnitude[i+1, j-1]]
            elif 67.5 <= angle < 112.5:
                neighbors = [gradient_magnitude[i-1, j], gradient_magnitude[i+1, j]]
            else:
                neighbors = [gradient_magnitude[i-1, j-1], gradient_magnitude[i+1, j+1]]

            if magnitude >= max(neighbors[0], neighbors[1]):
                suppressed[i, j] = magnitude

    return suppressed


def double_thresholding(image, low_ratio=0.5, high_ratio=1.0):
    """
    Apply double thresholding to classify edges as strong, weak, or non-edges

    Args:
        image: Input edge image (from non-max suppression)
        low_ratio: Low threshold ratio
        high_ratio: High threshold ratio

    Returns:
        Image with classified edges (strong=255, weak=75, non-edge=0)
    """
    high_threshold = image.max() * high_ratio
    low_threshold = high_threshold * low_ratio

    thresholded = np.zeros_like(image)

    strong_i, strong_j = np.where(image >= high_threshold)
    weak_i, weak_j = np.where((image >= low_threshold) & (image < high_threshold))

    thresholded[strong_i, strong_j] = 255
    thresholded[weak_i, weak_j] = 75

    return thresholded


def edge_tracking(image):
    """
    Perform edge tracking by hysteresis to convert weak edges to strong if connected to strong edges

    Args:
        image: Input image with classified edges (strong=255, weak=75, non-edge=0)

    Returns:
        Final edge image
    """
    height, width = image.shape

    tracked = np.copy(image)

    strong = 255
    weak = 75

    weak_i, weak_j = np.where(image == weak)
    weak_indices = list(zip(weak_i, weak_j))

    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while weak_indices:
        i, j = weak_indices.pop(0)

        is_connected_to_strong = False
        for di, dj in neighbors:
            ni, nj = i + di, j + dj
            if 0 <= ni < height and 0 <= nj < width and tracked[ni, nj] == strong:
                is_connected_to_strong = True
                break

        if is_connected_to_strong:
            tracked[i, j] = strong

            for di, dj in neighbors:
                ni, nj = i + di, j + dj
                if 0 <= ni < height and 0 <= nj < width and tracked[ni, nj] == weak and (ni, nj) not in weak_indices:
                    weak_indices.append((ni, nj))
        else:
            tracked[i, j] = 0

    return tracked


def normalize_image(image):
    """
    Normalize image values to [0, 255]

    Args:
        image: Input image

    Returns:
        Normalized image
    """
    if image.max() > 0:
        return (image / image.max() * 255).astype(np.uint8)
    return image.astype(np.uint8)


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    image_path = os.path.join(INPUT_DIR, "building.jpg")
    original_image = cv2.imread(image_path)

    if original_image is None:
        print(f"Error: Could not read image from {image_path}")
        return

    print(f"Processing image: {image_path}")
    print(f"Image shape: {original_image.shape}")

    # Step 1: Convert to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply Gaussian smoothing with sigma=1.4
    print("Step 1: Applying Gaussian smoothing...")
    smoothed_image = gaussian_smoothing(gray_image, sigma=1.4)
    smoothed_image = normalize_image(smoothed_image)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2d_canny1.jpg"), smoothed_image)
    print(f"Saved Gaussian smoothed image to {os.path.join(OUTPUT_DIR, 'ex2d_canny1.jpg')}")

    # Step 3: Apply Sobel operator
    print("Step 2: Calculating gradient magnitude using Sobel operator...")
    gradient_magnitude, gradient_direction = sobel_operator(smoothed_image)
    magnitude_image = normalize_image(gradient_magnitude)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2d_canny2.jpg"), magnitude_image)
    print(f"Saved gradient magnitude image to {os.path.join(OUTPUT_DIR, 'ex2d_canny2.jpg')}")

    # Step 4: Non-maximum suppression
    print("Step 3: Applying non-maximum suppression...")
    suppressed_image = non_max_suppression(gradient_magnitude, gradient_direction)
    suppressed_image = normalize_image(suppressed_image)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2d_canny3.jpg"), suppressed_image)
    print(f"Saved non-maximum suppression image to {os.path.join(OUTPUT_DIR, 'ex2d_canny3.jpg')}")

    # Step 5: Double thresholding
    print("Step 4: Applying double thresholding...")
    thresholded_image = double_thresholding(suppressed_image, low_ratio=0.5, high_ratio=1.0)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2d_canny4.jpg"), thresholded_image)
    print(f"Saved double thresholded image to {os.path.join(OUTPUT_DIR, 'ex2d_canny4.jpg')}")

    # Step 6: Edge tracking by hysteresis
    print("Step 5: Performing edge tracking...")
    final_image = edge_tracking(thresholded_image)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2d_canny5.jpg"), final_image)
    print(f"Saved edge tracked image to {os.path.join(OUTPUT_DIR, 'ex2d_canny5.jpg')}")

    # Step 7: Compare with OpenCV's Canny implementation
    print("Step 6: Comparing with OpenCV's Canny implementation...")
    low_threshold = int(smoothed_image.max() * 0.5)
    high_threshold = int(smoothed_image.max())
    opencv_canny = cv2.Canny(smoothed_image, low_threshold, high_threshold)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2d_canny6.jpg"), opencv_canny)
    print(f"Saved OpenCV Canny edge detection image to {os.path.join(OUTPUT_DIR, 'ex2d_canny6.jpg')}")

    print("Canny edge detection completed successfully!")


if __name__ == "__main__":
    main()
