import cv2
import numpy as np
import argparse

def main(args):
    """Main function to load frame, detect lines, and display."""
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.frame >= total_frames:
        print(f"Error: Frame {args.frame} is out of bounds. Video has {total_frames} frames.")
        cap.release()
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Error: Could not read frame {args.frame}.")
        return

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create a Line Segment Detector object
    lsd = cv2.createLineSegmentDetector(0)

    # Detect lines in the image
    lines, width, prec, nfa = lsd.detect(gray)

    # Draw the detected lines on the original frame
    if lines is not None:
        drawn_frame = lsd.drawSegments(frame, lines)
    else:
        drawn_frame = frame

    # Display the result
    cv2.imshow("LSD Line Detection", drawn_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="Detect lines in a video frame using LSD.")
    parser.add_argument("--video", type=str, default="video.mp4", help="Path to the video file.")
    parser.add_argument("--frame", type=int, default=2000, help="Frame number to load.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
