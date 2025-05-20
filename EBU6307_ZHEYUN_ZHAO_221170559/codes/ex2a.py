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


def gaussian_filter(image, kernel_size, sigma):
    """
    Apply Gaussian filter to an image

    Args:
        image: Input grayscale image
        kernel_size: Size of the Gaussian kernel
        sigma: Standard deviation of the Gaussian

    Returns:
        Filtered image
    """
    kernel = gaussian_kernel(kernel_size, sigma)

    return convolution2d(image, kernel)


def main():
    """
    Main function to process and filter an image using Gaussian filters.

    This function performs the following steps:
    1. Checks if the output directory exists; creates it if it does not.
    2. Checks for the existence of a grayscale image ('lena_gray.jpg') in the input directory.
        - If not found, converts the color image ('lena.jpg') to grayscale and saves it.
        - If found, loads the grayscale image.
    3. Applies Gaussian filters with different kernel sizes and sigma values to the grayscale image.
    4. Saves the filtered images to the output directory with filenames indicating the kernel size and sigma used.

    Dependencies:
         - os
         - cv2
         - numpy as np
         - gaussian_filter (custom or imported function)

    Assumptions:
         - INPUT_DIR and OUTPUT_DIR are defined globally.
         - The input directory contains 'lena.jpg' if 'lena_gray.jpg' is not present.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    image_path = os.path.join(INPUT_DIR, "lena_gray.jpg")

    if not os.path.exists(image_path):
        color_image = cv2.imread(os.path.join(INPUT_DIR, "lena.jpg"))
        if color_image is None:
            return
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(image_path, gray_image)
    else:
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray_image is None:
            return

    kernel_sizes = [5, 21]
    sigmas = [1, 10]

    for size in kernel_sizes:
        for sigma in sigmas:
            filtered_img = gaussian_filter(gray_image, size, sigma)

            filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)

            output_filename = os.path.join(OUTPUT_DIR, f"ex2a_gf_{size}_{sigma}.jpg")
            cv2.imwrite(output_filename, filtered_img)


if __name__ == "__main__":
    print('ex2a...')
    main()
