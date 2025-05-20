import cv2
import numpy as np


def downsampling(frame, factor):
    return frame[::factor, ::factor]


def upsampling(frame, factor):
    """
    Upsamples a 2D image (frame) using bilinear interpolation by a given integer factor.

    Parameters:
        frame (np.ndarray): 2D numpy array representing the grayscale image to be upsampled.
        factor (int): The upsampling factor (must be greater than 1).

    Returns:
        np.ndarray: The upsampled image as a 2D numpy array of type uint8.

    Raises:
        ValueError: If the upsampling factor is less than or equal to 1.

    Notes:
        - The function uses bilinear interpolation to compute the pixel values of the upsampled image.
        - The input frame must be a 2D numpy array (grayscale image).
    """
    if factor <= 1:
        raise ValueError("Factor must be greater than 1")

    height, width = frame.shape
    new_height, new_width = height * factor, width * factor

    y_indices, x_indices = np.meshgrid(np.arange(new_height), np.arange(new_width), indexing="ij")

    src_y = y_indices / factor
    src_x = x_indices / factor

    y0 = np.floor(src_y).astype(int)
    x0 = np.floor(src_x).astype(int)
    y1 = np.clip(y0 + 1, 0, height - 1)
    x1 = np.clip(x0 + 1, 0, width - 1)

    dy = src_y - y0
    dx = src_x - x0

    upsampled_frame = (
        (1 - dx) * (1 - dy) * frame[y0, x0] +
        dx * (1 - dy) * frame[y0, x1] +
        (1 - dx) * dy * frame[y1, x0] +
        dx * dy * frame[y1, x1]
    )

    return upsampled_frame.astype(np.uint8)


def intensity_downscaling(frame, n):
    level = 256 // n
    return (frame / level).astype(np.uint8) * level


print('ex1a...')

# Lena
im = cv2.imread('../inputs/lena_gray.jpg', cv2.IMREAD_GRAYSCALE)

im_1 = downsampling(im.copy(), 4)
cv2.imwrite('../results/ex1a_downsample_spatial_4.jpg', im_1)

im_1 = upsampling(im_1, 4)
cv2.imwrite('../results/ex1a_upsample_spatial_4.jpg', im_1)

im_2 = intensity_downscaling(im.copy(), 4)
cv2.imwrite('../results/ex1a_downsample_quant_4.jpg', im_2)

im_3 = intensity_downscaling(im.copy(), 16)
cv2.imwrite('../results/ex1a_downsample_quant_16.jpg', im_3)
