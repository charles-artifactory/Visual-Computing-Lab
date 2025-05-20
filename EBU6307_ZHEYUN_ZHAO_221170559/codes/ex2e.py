import numpy as np
import cv2
import os

INPUT_DIR = "../inputs/"
OUTPUT_DIR = "../results/"


def apply_fourier_transform(image):
    """
    Apply Fourier transform and shift the zero-frequency component to the center.

    Args:
        image: Input image.

    Returns:
        fft_image: Image after Fourier transform.
        fft_shift: Image after shifting the zero-frequency component to the center.
    """
    fft_image = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft_image)
    return fft_image, fft_shift


def inverse_fourier_transform(fft_shift):
    """
    Apply inverse Fourier transform to restore the image.

    Args:
        fft_shift: Shifted Fourier transformed image.

    Returns:
        restored_image: Restored image after inverse transform.
    """
    ifft_shift = np.fft.ifftshift(fft_shift)
    restored_image = np.fft.ifft2(ifft_shift).real
    return restored_image


def high_pass_filter(fft_shift, cutoff):
    """
    High-pass filter that sets the center region (low frequency) to zero.

    Args:
        fft_shift: Shifted Fourier transformed image.
        cutoff: Cutoff frequency.

    Returns:
        filtered: Image after high-pass filtering.
    """
    rows, cols = fft_shift.shape
    crow, ccol = rows // 2, cols // 2
    filtered = fft_shift.copy()
    filtered[crow-cutoff:crow+cutoff, ccol-cutoff:ccol+cutoff] = 0
    return filtered


def low_pass_filter(fft_shift, cutoff):
    """
    Low-pass filter that only keeps the center region (low frequency).

    Args:
        fft_shift: Shifted Fourier transformed image.
        cutoff: Cutoff frequency.

    Returns:
        filtered: Image after low-pass filtering.
    """
    rows, cols = fft_shift.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow-cutoff:crow+cutoff, ccol-cutoff:ccol+cutoff] = 1
    filtered = fft_shift * mask
    return filtered


def display_spectrum(fft_shift):
    """
    Display the spectrum of the Fourier transform.

    Args:
        fft_shift: Shifted Fourier transformed image.

    Returns:
        magnitude_spectrum: Visualized spectrum image.
    """
    magnitude_spectrum = 20 * np.log(np.abs(fft_shift) + 1)
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return magnitude_spectrum


def main():
    """
    Main function to read image, apply Fourier transform, filtering, and save results.
    """
    image_path = os.path.join(INPUT_DIR, "cat.png")
    image = cv2.imread(image_path, 0)
    if image is None:
        return
    fft_image, fft_shift = apply_fourier_transform(image)
    fourier_spectrum = display_spectrum(fft_shift)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2e_fourier.jpg"), fourier_spectrum)
    restored_image = inverse_fourier_transform(fft_shift)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2e_restored.jpg"), restored_image)
    low_pass = low_pass_filter(fft_shift, 30)
    low_pass_image = inverse_fourier_transform(low_pass)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2e_lowpass_30.jpg"), low_pass_image)
    high_pass = high_pass_filter(fft_shift, 30)
    high_pass_image = inverse_fourier_transform(high_pass)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2e_highpass_30.jpg"), high_pass_image)


if __name__ == "__main__":
    print('ex2e...')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()
