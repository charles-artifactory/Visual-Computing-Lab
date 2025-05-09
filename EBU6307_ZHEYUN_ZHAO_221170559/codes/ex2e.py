import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

INPUT_DIR = "../inputs/"
OUTPUT_DIR = "../results/"


def apply_fourier_transform(image):
    """
    Apply Fourier Transform to an image

    Args:
        image: Input grayscale image

    Returns:
        Tuple containing (fft_image, fft_shift)
    """
    fft_image = np.fft.fft2(image)

    fft_shift = np.fft.fftshift(fft_image)

    return fft_image, fft_shift


def inverse_fourier_transform(fft_shift):
    """
    Apply Inverse Fourier Transform to restore the image

    Args:
        fft_shift: Shifted Fourier Transform result

    Returns:
        Restored image
    """
    ifft_shift = np.fft.ifftshift(fft_shift)

    restored_image = np.fft.ifft2(ifft_shift).real

    return restored_image


def high_pass_filter(fft_shift, cutoff_freq):
    """
    Apply high-pass filter in frequency domain

    Args:
        fft_shift: Shifted Fourier Transform result
        cutoff_freq: Cutoff frequency for the filter

    Returns:
        Filtered frequency domain image
    """
    rows, cols = fft_shift.shape
    crow, ccol = rows // 2, cols // 2

    fft_filtered = fft_shift.copy()

    mask = np.ones((rows, cols), np.uint8)
    mask[crow-cutoff_freq:crow+cutoff_freq, ccol-cutoff_freq:ccol+cutoff_freq] = 0

    fft_filtered = fft_filtered * mask

    return fft_filtered


def low_pass_filter(fft_shift, cutoff_freq):
    """
    Apply low-pass filter in frequency domain

    Args:
        fft_shift: Shifted Fourier Transform result
        cutoff_freq: Cutoff frequency for the filter

    Returns:
        Filtered frequency domain image
    """
    rows, cols = fft_shift.shape
    crow, ccol = rows // 2, cols // 2

    fft_filtered = np.zeros_like(fft_shift)

    mask = np.zeros((rows, cols), np.uint8)
    mask[crow-cutoff_freq:crow+cutoff_freq, ccol-cutoff_freq:ccol+cutoff_freq] = 1

    fft_filtered = fft_shift * mask

    return fft_filtered


def visualize_fourier_spectrum(fft_shift):
    """
    Visualize Fourier spectrum for display

    Args:
        fft_shift: Shifted Fourier Transform result

    Returns:
        Logarithmic scale visualization of the Fourier spectrum
    """
    magnitude_spectrum = 20 * np.log(np.abs(fft_shift) + 1)

    magnitude_spectrum = np.clip(magnitude_spectrum, 0, 255)
    magnitude_spectrum = (magnitude_spectrum / magnitude_spectrum.max() * 255).astype(np.uint8)

    return magnitude_spectrum


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    image_path = os.path.join(INPUT_DIR, "cat.png")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return

    print(f"Processing image: {image_path}")
    print(f"Image shape: {image.shape}")

    # i. Apply Fourier Transform and visualize
    fft_image, fft_shift = apply_fourier_transform(image)
    fourier_spectrum = visualize_fourier_spectrum(fft_shift)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2e_fourier.jpg"), fourier_spectrum)
    print(f"Saved Fourier spectrum to {os.path.join(OUTPUT_DIR, 'ex2e_fourier.jpg')}")

    # ii. Apply Inverse Fourier Transform to restore the image
    restored_image = inverse_fourier_transform(fft_shift)
    restored_image = np.clip(restored_image, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2e_restored.jpg"), restored_image)
    print(f"Saved restored image to {os.path.join(OUTPUT_DIR, 'ex2e_restored.jpg')}")

    # iii. Apply High-Pass Filter with cutoff frequency of 30
    cutoff_freq = 30
    high_pass = high_pass_filter(fft_shift, cutoff_freq)
    high_pass_spectrum = visualize_fourier_spectrum(high_pass)

    # Restore high-pass filtered image
    high_pass_image = inverse_fourier_transform(high_pass)
    high_pass_image = np.clip(high_pass_image, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"ex2e_highpass_{cutoff_freq}.jpg"), high_pass_image)
    print(f"Saved high-pass filtered image to {os.path.join(OUTPUT_DIR, f'ex2e_highpass_{cutoff_freq}.jpg')}")

    # iv. Apply Low-Pass Filter with cutoff frequency of 30
    low_pass = low_pass_filter(fft_shift, cutoff_freq)
    low_pass_spectrum = visualize_fourier_spectrum(low_pass)

    # Restore low-pass filtered image
    low_pass_image = inverse_fourier_transform(low_pass)
    low_pass_image = np.clip(low_pass_image, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"ex2e_low pass_{cutoff_freq}.jpg"), low_pass_image)
    print(f"Saved low-pass filtered image to {os.path.join(OUTPUT_DIR, f'ex2e_low pass_{cutoff_freq}.jpg')}")


if __name__ == "__main__":
    main()
