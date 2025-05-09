import cv2
import os

INPUT_DIR = "../inputs/"
OUTPUT_DIR = "../results/"


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    video_path = os.path.join(INPUT_DIR, "ebu6304_chaplin.mp4")
    output_path = os.path.join(OUTPUT_DIR, "ex3c_meanshift_track_chaplinface.mp4")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")

    frames_to_process = min(fps * 4, total_frames)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frames_to_save = [1, 20, 40, 60, 90]

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame")
        cap.release()
        return

    x, y, w, h = 455, 150, 80, 80
    track_window = (x, y, w, h)

    roi = frame[y:y+h, x:x+w]

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    roi_hist = cv2.calcHist([roi_gray], [0], None, [256], [0, 256])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    initial_frame = frame.copy()
    cv2.rectangle(initial_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if 1 in frames_to_save:
        frame_save_path = os.path.join(OUTPUT_DIR, f"frame_1.png")
        cv2.imwrite(frame_save_path, initial_frame)
        print(f"Saved frame 1 to {frame_save_path}")

    frame_count = 0

    tracker = cv2.TrackerKCF_create()
    tracker.init(frame, (x, y, w, h))

    while frame_count < frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        success, box = tracker.update(frame)

        if success:
            x, y, w, h = [int(v) for v in box]
        else:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dst = cv2.calcBackProject([gray_frame], [0], roi_hist, [0, 256], 1)
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)
            x, y, w, h = track_window

        tracked_frame = frame.copy()
        cv2.rectangle(tracked_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        out.write(tracked_frame)

        if frame_count + 1 in frames_to_save:
            frame_save_path = os.path.join(OUTPUT_DIR, f"frame_{frame_count + 1}.png")
            cv2.imwrite(frame_save_path, tracked_frame)
            print(f"Saved frame {frame_count + 1} to {frame_save_path}")

        if frame_count % 10 == 0:
            print(f"Processed {frame_count}/{frames_to_process} frames")

    cap.release()
    out.release()

    print(f"Tracking completed. Output saved to {output_path}")


if __name__ == "__main__":
    main()
