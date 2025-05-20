import cv2
import numpy as np
import os

INPUT_DIR = "../inputs/"
OUTPUT_DIR = "../results/"
SAVE_FRAMES = [1, 20, 40, 60, 90]  # Frame indices to save (1-based)


def load_video(video_path):
    """
    Load a video file and return the VideoCapture object.
    Raises RuntimeError if the video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    return cap


def get_first_frame(cap):
    """
    Read and return the first frame from the video capture.
    Raises RuntimeError if the frame cannot be read.
    """
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Unable to read first frame")
    return frame


def initialize_tracker(frame, rect):
    """
    Initialize tracking parameters.
    Returns the grayscale ROI histogram and the initial tracking window.
    """
    x, y, w, h = rect
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = gray[y:y+h, x:x+w]
    roi_hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    return roi_hist, rect


def create_video_writer(output_path, fps, width, height):
    """
    Create and return a VideoWriter object for saving output video.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def process_video(cap, roi_hist, track_window, out, save_frames, output_dir, max_frames):
    """
    Process the video frame by frame, apply mean shift tracking,
    save specified frames, and write output video.
    """
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    frame_idx = 1

    while frame_idx <= max_frames:
        if frame_idx == 1:
            curr_frame = get_first_frame(cap)
        else:
            ret, curr_frame = cap.read()
            if not ret:
                break

        gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        backproj = cv2.calcBackProject([gray], [0], roi_hist, [0, 256], 1)
        _, track_window = cv2.meanShift(backproj, track_window, term_crit)
        x, y, w, h = track_window

        out_frame = cv2.rectangle(curr_frame.copy(), (x, y), (x+w, y+h), (0, 255, 0), 2)
        out.write(out_frame)

        if frame_idx in save_frames:
            save_path = os.path.join(output_dir, f"frame_{frame_idx}.png")
            cv2.imwrite(save_path, out_frame)

        frame_idx += 1


def main():
    """
    Main function to perform mean shift tracking on a video and save results.
    """
    video_path = os.path.join(INPUT_DIR, "ebu6304_chaplin.mp4")
    output_path = os.path.join(OUTPUT_DIR, "ex3c_meanshift_track_chaplinface.mp4")

    cap = load_video(video_path)
    first_frame = get_first_frame(cap)

    # Initial tracking window (x, y, w, h)
    rect = (471, 168, 49, 57)
    roi_hist, track_window = initialize_tracker(first_frame, rect)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(fps * 4)

    out = create_video_writer(output_path, fps, width, height)

    # Reset video to first frame for processing
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    process_video(cap, roi_hist, track_window, out, SAVE_FRAMES, OUTPUT_DIR, max_frames)

    cap.release()
    out.release()
    print(f"Saved tracked video to {output_path}")
    print(f"Saved frames: {SAVE_FRAMES} to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
