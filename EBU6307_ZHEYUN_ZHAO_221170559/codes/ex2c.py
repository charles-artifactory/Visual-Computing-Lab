import cv2
import os

INPUT_DIR = "../inputs/"
OUTPUT_DIR = "../results/"


def main():
    """
    Main function to perform SIFT keypoint detection and feature matching on two images.
    - Reads two input images.
    - Detects SIFT keypoints and descriptors.
    - Visualizes and saves keypoints.
    - Performs brute-force matching and saves visualizations of best and worst matches.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    img1_path = os.path.join(INPUT_DIR, "sift_input1.jpg")
    img2_path = os.path.join(INPUT_DIR, "sift_input2.jpg")

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        return

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    img1_keypoints = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_keypoints = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2c_sift_input1.jpg"), img1_keypoints)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2c_sift_input2.jpg"), img2_keypoints)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    best_matches = matches[:10]
    worst_matches = matches[-10:]

    img_best_matches = cv2.drawMatches(img1, kp1, img2, kp2, best_matches, None,
                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    img_worst_matches = cv2.drawMatches(img1, kp1, img2, kp2, worst_matches, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2c_matches_most10.jpg"), img_best_matches)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ex2c_matches_least10.jpg"), img_worst_matches)


if __name__ == "__main__":
    print('ex2c...')
    main()
