import cv2
import numpy as np
import argparse

def draw_hough_line(frame, rho, theta_deg):
    """Draws a line on a frame given rho and theta (in degrees)."""
    h, w = frame.shape[:2]
    theta_rad = np.deg2rad(theta_deg)
    
    a = np.cos(theta_rad)
    b = np.sin(theta_rad)
    x0 = a * rho
    y0 = b * rho
    
    # Derive two points on the line to draw it
    # We create a line 2000 pixels long, which is enough to span any typical video frame
    x1 = int(x0 + 2000 * (-b))
    y1 = int(y0 + 2000 * (a))
    x2 = int(x0 - 2000 * (-b))
    y2 = int(y0 - 2000 * (a))
    
    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return frame

def main(args):
    """Main function to load frame and draw the specified line."""
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.frame >= total_frames:
        print(f"Error: Frame {args.frame} is out of bounds. Video has {total_frames} frames.")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame {args.frame}.")
        return
    
    print(f"Drawing line with Rho = {args.rho} and Theta = {args.theta} degrees.")
    
    frame_with_line = draw_hough_line(frame, args.rho, args.theta)
    
    window_name = "Hough Line Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while True:
        cv2.imshow(window_name, frame_with_line)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="Draw a single Hough line on a video frame.")
    parser.add_argument("--video", type=str, default="video.mp4", help="Path to the video file.")
    parser.add_argument("--frame", type=int, default=2000, help="Frame number to load.")
    parser.add_argument("--rho", type=float, default=100.0, help="The 'rho' parameter of the line (distance from origin).")
    parser.add_argument("--theta", type=float, default=45.0, help="The 'theta' parameter of the line (angle in degrees).")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
