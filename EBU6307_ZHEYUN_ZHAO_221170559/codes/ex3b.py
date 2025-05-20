import numpy as np
import cv2
import os

INPUT_DIR = "../inputs/"
OUTPUT_DIR = "../results/"


def compute_adaptive_weight(p_intensity, q_intensity, p_pos, q_pos, gamma_c=10.0, gamma_p=17.5):
    """
    Compute the adaptive weight between two pixels based on intensity and spatial proximity.

    Args:
        p_intensity (float): Intensity of the center pixel.
        q_intensity (float): Intensity of the neighboring pixel.
        p_pos (tuple): Position (y, x) of the center pixel.
        q_pos (tuple): Position (y, x) of the neighboring pixel.
        gamma_c (float): Parameter controlling the influence of intensity similarity.
        gamma_p (float): Parameter controlling the influence of spatial proximity.

    Returns:
        float: Weight value.
    """
    intensity_diff = np.abs(float(p_intensity) - float(q_intensity))
    color_weight = np.exp(-intensity_diff / gamma_c)
    spatial_dist = np.sqrt((p_pos[0] - q_pos[0])**2 + (p_pos[1] - q_pos[1])**2)
    spatial_weight = np.exp(-spatial_dist / gamma_p)
    return color_weight * spatial_weight


def compute_disparity_map_adaptive(img_left, img_right, window_size, max_disparity=60):
    """
    Compute disparity map using adaptive support-weight based stereo matching.

    Args:
        img_left (np.ndarray): Left image.
        img_right (np.ndarray): Right image.
        window_size (int): Size of the window for cost aggregation.
        max_disparity (int): Maximum disparity value to search.

    Returns:
        np.ndarray: Disparity map.
    """
    if len(img_left.shape) == 3:
        img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    h, w = img_left.shape
    pad = window_size // 2
    disparity_map = np.zeros((h, w), dtype=np.float32)
    img_left_pad = np.pad(img_left, pad, mode='edge')
    img_right_pad = np.pad(img_right, pad, mode='edge')

    for y in range(h):
        for x in range(w):
            min_cost = float('inf')
            best_disparity = 0
            p_pos = (y, x)
            p_intensity = img_left[y, x]
            for d in range(max_disparity):
                if x - d < 0:
                    continue
                weighted_cost = 0.0
                sum_weights = 0.0
                for wy in range(-pad, pad + 1):
                    for wx in range(-pad, pad + 1):
                        qy = y + wy
                        qx = x + wx
                        if qy < 0 or qy >= h or qx < 0 or qx >= w:
                            continue
                        if qx - d < 0:
                            continue
                        q_pad_pos = (qy + pad, qx + pad)
                        q_pos = (qy, qx)
                        q_intensity = img_left_pad[q_pad_pos[0], q_pad_pos[1]]
                        left_val = float(img_left_pad[q_pad_pos[0], q_pad_pos[1]])
                        right_val = float(img_right_pad[q_pad_pos[0], q_pad_pos[1] - d])
                        cost = (left_val - right_val) ** 2
                        weight = compute_adaptive_weight(p_intensity, q_intensity, p_pos, q_pos)
                        weighted_cost += weight * cost
                        sum_weights += weight
                if sum_weights > 0:
                    aggregated_cost = weighted_cost / sum_weights
                else:
                    aggregated_cost = float('inf')
                if aggregated_cost < min_cost:
                    min_cost = aggregated_cost
                    best_disparity = d
            disparity_map[y, x] = best_disparity
    disparity_map = (disparity_map * 255 / max_disparity).astype(np.uint8)
    return disparity_map


def main():
    """
    Main function to read stereo images, compute disparity maps using adaptive support-weight, and save the results.
    """
    img_left = cv2.imread(os.path.join(INPUT_DIR, "teddy_im2.png"))
    img_right = cv2.imread(os.path.join(INPUT_DIR, "teddy_im6.png"))
    if img_left is None or img_right is None:
        return
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    window_sizes = [3, 11]
    for w_size in window_sizes:
        disparity_map = compute_disparity_map_adaptive(img_left, img_right, w_size)
        output_filename = f"ex3b_aw_{w_size}.png"
        cv2.imwrite(os.path.join(OUTPUT_DIR, output_filename), disparity_map)


if __name__ == "__main__":
    print('ex3b...')
    main()
