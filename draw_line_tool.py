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

def calculate_color_std_dev(frame, rho, theta_deg, num_samples):
    """Calculates the standard deviation of colors along a line in the frame."""
    h, w = frame.shape[:2]
    theta_rad = np.deg2rad(theta_deg)
    
    a = np.cos(theta_rad)
    b = np.sin(theta_rad)
    x0 = a * rho
    y0 = b * rho
    
    # Points far away to define the line
    x1 = int(x0 + 2000 * (-b))
    y1 = int(y0 + 2000 * (a))
    x2 = int(x0 - 2000 * (-b))
    y2 = int(y0 - 2000 * (a))

    # Clip the line to the frame boundaries
    rect = (0, 0, w, h)
    inside, p1, p2 = cv2.clipLine(rect, (x1, y1), (x2, y2))

    if inside:
        # Generate sample points along the clipped line
        x_coords = np.linspace(p1[0], p2[0], num_samples, dtype=int)
        y_coords = np.linspace(p1[1], p2[1], num_samples, dtype=int)

        # Ensure coordinates are within frame bounds
        x_coords = np.clip(x_coords, 0, w - 1)
        y_coords = np.clip(y_coords, 0, h - 1)
        
        # Convert frame to grayscale for simplicity
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get pixel values at sample points
        pixel_values = gray_frame[y_coords, x_coords]
        
        # Calculate and print standard deviation
        std_dev = np.std(pixel_values)
        print(f"Standard deviation of colors along the line ({num_samples} samples): {std_dev:.2f}")
    else:
        print("Line is outside the frame viewport.")

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
    
    if args.num_samples > 0:
        calculate_color_std_dev(frame, args.rho, args.theta, args.num_samples)
    
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
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples along the line to calculate color std deviation.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
