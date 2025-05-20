import numpy as np
import cv2
import os

INPUT_DIR = "../inputs/"
OUTPUT_DIR = "../results/"


def ensure_dir_exists(dir_path):
    """
    Ensure the output directory exists.
    Args:
        dir_path (str): Path to the directory.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def convolve2d(image, kernel):
    """
    Manually implement 2D convolution operation.
    Args:
        image (np.ndarray): Input 2D image array.
        kernel (np.ndarray): Convolution kernel.
    Returns:
        np.ndarray: Convolved image.
    """
    img_height, img_width = image.shape
    k_height, k_width = kernel.shape
    pad_height = k_height // 2
    pad_width = k_width // 2
    output = np.zeros_like(image, dtype=np.float64)
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), 'reflect')
    for i in range(img_height):
        for j in range(img_width):
            patch = padded_image[i:i+k_height, j:j+k_width]
            output[i, j] = np.sum(patch * kernel)
    return output


def apply_anti_aliasing(image, kernel_size=5):
    """
    Apply mean filter for anti-aliasing.
    Args:
        image (np.ndarray): Input image (grayscale or color).
        kernel_size (int): Size of the mean filter kernel.
    Returns:
        np.ndarray: Anti-aliased image.
    """
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    result = np.zeros_like(image, dtype=np.float64)
    if len(image.shape) == 3:
        for c in range(image.shape[2]):
            result[:, :, c] = convolve2d(image[:, :, c], kernel)
    else:
        result = convolve2d(image, kernel)
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_sharpening(image, alpha=0.5):
    """
    Apply sharpening filter to enhance edges.
    Args:
        image (np.ndarray): Input image (grayscale or color).
        alpha (float): Sharpening strength.
    Returns:
        np.ndarray: Sharpened image.
    """
    laplacian_kernel = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])
    result = np.zeros_like(image, dtype=np.float64)
    if len(image.shape) == 3:
        for c in range(image.shape[2]):
            high_pass = convolve2d(image[:, :, c], laplacian_kernel)
            result[:, :, c] = image[:, :, c] + alpha * high_pass
    else:
        high_pass = convolve2d(image, laplacian_kernel)
        result = image + alpha * high_pass
    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    """
    Main function to apply anti-aliasing and sharpening filters to an image.
    """
    ensure_dir_exists(OUTPUT_DIR)
    input_image_path = os.path.join(INPUT_DIR, "Squares.jpg")
    image = cv2.imread(input_image_path)
    if image is None:
        return
    anti_aliased_image = apply_anti_aliasing(image)
    anti_aliased_path = os.path.join(OUTPUT_DIR, "ex2b_Alias.jpg")
    cv2.imwrite(anti_aliased_path, anti_aliased_image)
    sharpened_image = apply_sharpening(anti_aliased_image)
    sharpened_path = os.path.join(OUTPUT_DIR, "ex2b_Sharp.jpg")
    cv2.imwrite(sharpened_path, sharpened_image)


if __name__ == "__main__":
    print('ex2b...')
    main()
