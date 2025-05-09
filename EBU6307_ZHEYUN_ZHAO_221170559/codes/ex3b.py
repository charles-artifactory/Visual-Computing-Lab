import numpy as np
import cv2
import os
import math

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


def compute_adaptive_weights(image, window_size, gamma_c=10.0, gamma_p=10.0):
    """
    Compute adaptive support-weights for all pixels in the image

    Args:
        image: Input image
        window_size: Size of the window
        gamma_c: Parameter for color similarity
        gamma_p: Parameter for spatial proximity

    Returns:
        Dictionary of weight maps for each pixel position
    """
    height, width = image.shape

    half_window = window_size // 2

    weights = {}

    for y in range(height):
        for x in range(width):
            center_value = float(image[y, x])

            weight_map = np.zeros((window_size, window_size), dtype=np.float32)

            for wy in range(window_size):
                for wx in range(window_size):
                    ny = y - half_window + wy
                    nx = x - half_window + wx

                    if 0 <= ny < height and 0 <= nx < width:
                        color_diff = abs(float(image[ny, nx]) - center_value)

                        spatial_dist = math.sqrt((nx - x) ** 2 + (ny - y) ** 2)

                        weight = math.exp(-color_diff / gamma_c) * math.exp(-spatial_dist / gamma_p)
                    else:
                        weight = 0.0

                    weight_map[wy, wx] = weight

            weights[(y, x)] = weight_map

    return weights


def aggregate_costs_adaptive(cost_volume, image, window_size, gamma_c=10.0, gamma_p=10.0):
    """
    Aggregate costs using adaptive support-weights

    Args:
        cost_volume: 3D cost volume with matching costs
        image: Input image (left image for weights calculation)
        window_size: Size of the aggregation window
        gamma_c: Parameter for color similarity
        gamma_p: Parameter for spatial proximity

    Returns:
        Aggregated cost volume
    """
    height, width, num_disparities = cost_volume.shape

    aggregated_cost = np.zeros_like(cost_volume)

    half_window = window_size // 2

    print("Computing adaptive weights...")
    weights = compute_adaptive_weights(image, window_size, gamma_c, gamma_p)

    print("Aggregating costs...")
    for d in range(num_disparities):
        for y in range(height):
            for x in range(width):
                weight_map = weights[(y, x)]

                weighted_sum = 0.0
                weight_sum = 0.0

                for wy in range(window_size):
                    for wx in range(window_size):
                        ny = y - half_window + wy
                        nx = x - half_window + wx

                        if 0 <= ny < height and 0 <= nx < width:
                            weight = weight_map[wy, wx]

                            weighted_sum += weight * cost_volume[ny, nx, d]
                            weight_sum += weight

                if weight_sum > 0:
                    aggregated_cost[y, x, d] = weighted_sum / weight_sum
                else:
                    aggregated_cost[y, x, d] = cost_volume[y, x, d]

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


def adaptive_stereo_matching(left_img, right_img, window_size, max_disparity):
    """
    Perform adaptive support-weight based stereo matching

    Args:
        left_img: Left stereo image
        right_img: Right stereo image
        window_size: Size of the aggregation window
        max_disparity: Maximum disparity value to consider

    Returns:
        Disparity map
    """
    print(f"Computing adaptive support-weight stereo matching with window size {window_size}x{window_size}...")

    # Step 1: Compute matching cost for all disparities
    cost_volume = compute_matching_cost(left_img, right_img, max_disparity)

    # Step 2: Aggregate costs using adaptive support-weights
    aggregated_cost = aggregate_costs_adaptive(cost_volume, left_img, window_size)

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
        disparity_map = adaptive_stereo_matching(left_img, right_img, window_size, max_disparity)

        normalized_disparity = (disparity_map * (255 / max_disparity)).astype(np.uint8)

        output_filename = os.path.join(OUTPUT_DIR, f"ex3b_aw_{window_size}.png")
        cv2.imwrite(output_filename, normalized_disparity)
        print(f"Saved disparity map with window size {window_size}x{window_size} to {output_filename}")


if __name__ == "__main__":
    main()
