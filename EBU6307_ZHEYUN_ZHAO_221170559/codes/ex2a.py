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
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    image_path = os.path.join(INPUT_DIR, "lena_gray.jpg")

    if not os.path.exists(image_path):
        print(f"Grayscale image not found at {image_path}, converting from color image...")
        color_image = cv2.imread(os.path.join(INPUT_DIR, "lena.jpg"))
        if color_image is None:
            print("Error: Could not read color image")
            return
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(image_path, gray_image)
        print(f"Saved grayscale image to {image_path}")
    else:
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray_image is None:
            print(f"Error: Could not read grayscale image from {image_path}")
            return

    kernel_sizes = [5, 21]
    sigmas = [1, 10]

    for size in kernel_sizes:
        for sigma in sigmas:
            print(f"Applying Gaussian filter with kernel size = {size}, sigma = {sigma}")

            filtered_img = gaussian_filter(gray_image, size, sigma)

            filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)

            output_filename = os.path.join(OUTPUT_DIR, f"ex2a_gf_{size}_{sigma}.jpg")
            cv2.imwrite(output_filename, filtered_img)
            print(f"Saved result to {output_filename}")


if __name__ == "__main__":
    main()
