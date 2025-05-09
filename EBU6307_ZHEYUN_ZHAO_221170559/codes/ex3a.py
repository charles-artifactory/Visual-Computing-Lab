import numpy as np
import cv2
import os

INPUT_DIR = "../inputs/"
OUTPUT_DIR = "../results/"


def compute_matching_cost(left_img, right_img, max_disparity):
    """
    Compute the matching cost for all possible disparities

    Args:
        left_img: Left stereo image
        right_img: Right stereo image
        max_disparity: Maximum disparity value to consider

    Returns:
        3D cost volume with dimensions (height, width, max_disparity+1)
    """
    height, width = left_img.shape

    cost_volume = np.zeros((height, width, max_disparity + 1), dtype=np.float32)

    for d in range(max_disparity + 1):
        right_shifted = np.zeros_like(right_img)
        right_shifted[:, 0:width-d] = right_img[:, d:width]

        diff = (left_img.astype(np.float32) - right_shifted.astype(np.float32)) ** 2
        cost_volume[:, :, d] = diff

    return cost_volume


def aggregate_costs(cost_volume, window_size):
    """
    Aggregate costs using a window-based approach

    Args:
        cost_volume: 3D cost volume with matching costs
        window_size: Size of the aggregation window

    Returns:
        Aggregated cost volume
    """
    height, width, num_disparities = cost_volume.shape

    aggregated_cost = np.zeros_like(cost_volume)

    half_window = window_size // 2

    for d in range(num_disparities):
        for y in range(height):
            for x in range(width):
                y_start = max(0, y - half_window)
                y_end = min(height, y + half_window + 1)
                x_start = max(0, x - half_window)
                x_end = min(width, x + half_window + 1)

                aggregated_cost[y, x, d] = np.sum(cost_volume[y_start:y_end, x_start:x_end, d])

                actual_window_size = (y_end - y_start) * (x_end - x_start)
                aggregated_cost[y, x, d] /= actual_window_size

    return aggregated_cost


def compute_disparity(aggregated_cost):
    """
    Compute disparity map by finding minimum cost disparity

    Args:
        aggregated_cost: Aggregated cost volume

    Returns:
        Disparity map
    """
    disparity_map = np.argmin(aggregated_cost, axis=2)

    return disparity_map


def stereo_matching(left_img, right_img, window_size, max_disparity):
    """
    Perform window-based stereo matching

    Args:
        left_img: Left stereo image
        right_img: Right stereo image
        window_size: Size of the aggregation window
        max_disparity: Maximum disparity value to consider

    Returns:
        Disparity map
    """
    print(f"Computing stereo matching with window size {window_size}x{window_size}...")

    # Step 1: Compute matching cost for all disparities
    cost_volume = compute_matching_cost(left_img, right_img, max_disparity)

    # Step 2: Aggregate costs using window-based approach
    aggregated_cost = aggregate_costs(cost_volume, window_size)

    # Step 3: Compute disparity map
    disparity_map = compute_disparity(aggregated_cost)

    return disparity_map


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    left_img_path = os.path.join(INPUT_DIR, "teddy_im2.png")
    right_img_path = os.path.join(INPUT_DIR, "teddy_im6.png")

    left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

    if left_img is None or right_img is None:
        print(f"Error: Could not read stereo images")
        return

    print(f"Processing stereo images")
    print(f"Left image shape: {left_img.shape}, Right image shape: {right_img.shape}")

    max_disparity = 64

    window_sizes = [3, 11]

    for window_size in window_sizes:
        disparity_map = stereo_matching(left_img, right_img, window_size, max_disparity)

        normalized_disparity = (disparity_map * (255 / max_disparity)).astype(np.uint8)

        output_filename = os.path.join(OUTPUT_DIR, f"ex3a_w_{window_size}.png")
        cv2.imwrite(output_filename, normalized_disparity)
        print(f"Saved disparity map with window size {window_size}x{window_size} to {output_filename}")


if __name__ == "__main__":
    main()
