import numpy as np
import cv2
import os

INPUT_DIR = "../inputs/"
OUTPUT_DIR = "../results/"


def compute_disparity_map(img_left, img_right, window_size, max_disparity=60):
    """
    Compute disparity map using window-based stereo matching.

    Args:
        img_left (np.ndarray): Left image.
        img_right (np.ndarray): Right image.
        window_size (int): Size of the window for cost aggregation.
        max_disparity (int, optional): Maximum disparity value to search. Default is 60.

    Returns:
        np.ndarray: Disparity map normalized for visualization.
    """
    if len(img_left.shape) == 3:
        img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    h, w = img_left.shape
    pad = window_size // 2
    disparity_map = np.zeros((h, w), dtype=np.float32)

    img_left_pad = cv2.copyMakeBorder(img_left, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    img_right_pad = cv2.copyMakeBorder(img_right, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    for y in range(h):
        for x in range(w):
            min_cost = float('inf')
            best_disparity = 0

            window_left = img_left_pad[y:y + window_size, x:x + window_size]

            for d in range(max_disparity):
                if x - d < 0:
                    continue

                window_right = img_right_pad[y:y + window_size, (x - d):x - d + window_size]

                cost = np.sum((window_left.astype(np.float32) - window_right.astype(np.float32)) ** 2)

                if cost < min_cost:
                    min_cost = cost
                    best_disparity = d

            disparity_map[y, x] = best_disparity

    disparity_map = (disparity_map * 255 / max_disparity).astype(np.uint8)
    return disparity_map


def main():
    """
    Main function to read stereo images, compute disparity maps with different window sizes,
    and save the results to the output directory.
    """
    img_left = cv2.imread(os.path.join(INPUT_DIR, "teddy_im2.png"))
    img_right = cv2.imread(os.path.join(INPUT_DIR, "teddy_im6.png"))

    if img_left is None or img_right is None:
        return

    window_sizes = [3, 11]

    for w_size in window_sizes:
        disparity_map = compute_disparity_map(img_left, img_right, w_size)

        output_filename = f"ex3a_w_{w_size}.png"
        cv2.imwrite(os.path.join(OUTPUT_DIR, output_filename), disparity_map)


if __name__ == "__main__":
    print('ex3a...')
    main()
