import cv2
import numpy as np
import argparse
from line_detector import LineDetector

def main(args):
    """
    A simple script to step through video frames and visualize the raw output
    of the LineDetector.
    """
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = args.start_frame

    if frame_idx >= total_frames:
        print(f"Error: Start frame {frame_idx} is out of bounds. Video has {total_frames} frames.")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    # Use the tuned parameters that were giving the best (though still imperfect) results
    detector = LineDetector(hough_threshold=275, line_merge_dist=15)

    window_name = "Line Detector Debugger"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        print(f"\n--- Processing Frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1} ---")
        
        # We only care about the labeled frame and the raw lines for this debugger
        # The detect_lines function has been modified to return raw lines
        labeled_frame, raw_lines = detector.detect_lines(frame)

        if not raw_lines:
            print("No lines detected.")
        else:
            print("Raw detected lines (rho, theta):")
            for i, (rho, theta) in enumerate(raw_lines):
                print(f"  Line {i:02d}: rho={rho:8.2f}, theta={np.rad2deg(theta):7.2f} deg")

        cv2.imshow(window_name, labeled_frame)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord('d'):  # 'd' for next frame
            continue
        elif key == ord('a'): # 'a' for previous frame
            current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            # Subtract 2 because we want the frame before the *current* one, and the cap is already at the next frame
            prev_frame_idx = max(0, current_pos - 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, prev_frame_idx)


    cap.release()
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="Debug the LineDetector on a video.")
    parser.add_argument("--video", type=str, default="video.mp4", help="Path to the video file.")
    parser.add_argument("--start-frame", type=int, default=0, help="Frame number to start processing from.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
