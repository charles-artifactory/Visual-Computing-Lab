import numpy as np
import cv2
import os
import math

INPUT_DIR = "../inputs/"
OUTPUT_DIR = "../results/"


def gaussian_kernel(size, sigma):
    """
    Create a 2D Gaussian kernel for anti-aliasing

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


def laplacian_kernel():
    """
    Create a Laplacian kernel for image sharpening

    Returns:
        2D numpy array with the Laplacian kernel
    """
    return np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])


def convolution2d(image, kernel):
    """
    Apply 2D convolution on an image using the given kernel

    Args:
        image: Input image as numpy array
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


def anti_aliasing_filter(image, kernel_size=5, sigma=1.0):
    """
    Apply an anti-aliasing filter to an image using Gaussian blur

    Args:
        image: Input image
        kernel_size: Size of the Gaussian kernel
        sigma: Standard deviation of the Gaussian

    Returns:
        Anti-aliased image
    """
    kernel = gaussian_kernel(kernel_size, sigma)
    return convolution2d(image, kernel)


def sharpen_image(image, strength=1.0):
    """
    Sharpen an image using a Laplacian filter

    Args:
        image: Input image
        strength: Strength of sharpening effect

    Returns:
        Sharpened image
    """
    kernel = laplacian_kernel()
    edge_image = convolution2d(image, kernel)
    sharpened = image + strength * edge_image
    return np.clip(sharpened, 0, 255)


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Read the input image
    image_path = os.path.join(INPUT_DIR, "Squares.jpg")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return

    print(f"Processing image: {image_path}")
    print(f"Image shape: {image.shape}")

    # 1. Apply anti-aliasing filter
    print("Applying anti-aliasing filter...")
    anti_aliased = anti_aliasing_filter(image, kernel_size=5, sigma=1.0)
    anti_aliased = np.clip(anti_aliased, 0, 255).astype(np.uint8)

    # Save anti-aliased image
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2b_Alias.jpg"), anti_aliased)
    print(f"Saved anti-aliased image to {os.path.join(OUTPUT_DIR, 'ex2b_Alias.jpg')}")

    # 2. Apply image sharpening filter on anti-aliased image
    print("Applying sharpening filter on anti-aliased image...")
    sharpened = sharpen_image(anti_aliased, strength=1.0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    # Save sharpened image
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2b_Sharp.jpg"), sharpened)
    print(f"Saved sharpened image to {os.path.join(OUTPUT_DIR, 'ex2b_Sharp.jpg')}")


if __name__ == "__main__":
    main()
