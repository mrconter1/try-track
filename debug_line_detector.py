import cv2
import numpy as np
import argparse
from line_detector import LineDetector

def plot_rho_theta(lines, frame_shape, plot_size=(400, 600)):
    """
    Creates a 2D plot of lines in Rho-Theta space.
    X-axis: Theta (0-180 degrees), Y-axis: Rho
    """
    plot_img = np.zeros((plot_size[0], plot_size[1], 3), dtype=np.uint8)
    h, w = frame_shape[:2]
    max_rho = np.sqrt(h**2 + w**2)  # Max possible rho is the diagonal

    # Draw axes and labels
    cv2.line(plot_img, (0, plot_size[0] // 2), (plot_size[1], plot_size[0] // 2), (50, 50, 50), 1) # Rho=0 axis
    cv2.line(plot_img, (plot_size[1] // 2, 0), (plot_size[1] // 2, plot_size[0]), (50, 50, 50), 1) # Theta=90 deg axis
    cv2.putText(plot_img, "Theta (0-180 deg)", (plot_size[1] - 150, plot_size[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(plot_img, "Rho", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if not lines:
        return plot_img

    for rho, theta in lines:
        # Map theta (0, pi) to x-axis (0, plot_width)
        x = int((theta / np.pi) * plot_size[1])
        # Map rho (-max_rho, max_rho) to y-axis (0, plot_height)
        y = int(((rho + max_rho) / (2 * max_rho)) * plot_size[0])

        # Ensure points are within bounds
        x = np.clip(x, 0, plot_size[1] - 1)
        y = np.clip(y, 0, plot_size[0] - 1)
        
        cv2.circle(plot_img, (x, y), 3, (0, 255, 0), -1)

    return plot_img

def main(args):
    """
    A script to step through video frames and visualize the raw output
    of the LineDetector with real-time GUI controls for tuning.
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

    # Initial detector with default values that will be updated by sliders
    detector = LineDetector(
        canny_low=10,
        canny_high=50,
        hough_threshold=130,
        scale=args.scale
    )

    # --- GUI Setup ---
    window_name = "Line Detector Debugger"
    controls_window_name = "Controls"
    plot_window_name = "Rho-Theta Plot"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(controls_window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(plot_window_name, cv2.WINDOW_NORMAL)

    def nothing(x):
        pass

    # Create trackbars
    cv2.createTrackbar("Hough Threshold", controls_window_name, 130, 500, nothing)
    cv2.createTrackbar("Canny Low", controls_window_name, 10, 255, nothing)
    cv2.createTrackbar("Canny High", controls_window_name, 50, 255, nothing)
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the start frame.")
        cap.release()
        return

    while True:
        # --- Read GUI values ---
        hough_threshold = cv2.getTrackbarPos("Hough Threshold", controls_window_name)
        canny_low = cv2.getTrackbarPos("Canny Low", controls_window_name)
        canny_high = cv2.getTrackbarPos("Canny High", controls_window_name)

        # Enforce Canny logic: low threshold cannot be higher than high threshold
        canny_high = max(canny_high, canny_low + 1)
        cv2.setTrackbarPos("Canny High", controls_window_name, canny_high)

        # --- Update detector ---
        detector.hough_threshold = hough_threshold
        detector.canny_low = canny_low
        detector.canny_high = canny_high

        # --- Process and Display ---
        if frame is not None:
            current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            print(f"\rProcessing Frame: {current_frame_pos} | Hough: {hough_threshold}, Canny: {canny_low}/{canny_high}", end="")

            labeled_frame, raw_lines = detector.detect_lines_raw(frame.copy())
            plot_image = plot_rho_theta(raw_lines, frame.shape)
            
            cv2.imshow(window_name, labeled_frame)
            cv2.imshow(plot_window_name, plot_image)
        else:
            # Create a blank screen if no frame
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank_frame, "End of video", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow(window_name, blank_frame)


        # --- Handle Keyboard Input ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord('d'):  # 'd' for next frame
            ret, frame = cap.read()
            if not ret:
                print("\nEnd of video.")
                frame = None
        elif key == ord('a'): # 'a' for previous frame
            current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            prev_frame_idx = max(0, current_pos - 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, prev_frame_idx)
            ret, frame = cap.read()
            if not ret:
                frame = None


    cap.release()
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="Debug the LineDetector on a video.")
    parser.add_argument("--video", type=str, default="video.mp4", help="Path to the video file.")
    parser.add_argument("--start-frame", type=int, default=0, help="Frame number to start processing from.")
    parser.add_argument("--scale", type=float, default=0.5, help="Downscaling factor for processing.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
