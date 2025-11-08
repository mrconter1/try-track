import cv2
import numpy as np

class LineDetector:
    def __init__(self, canny_low=30, canny_high=100, hough_threshold=150, line_merge_dist=30, line_merge_angle=5):
        """Initialize line detector for grid lines"""
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.line_merge_dist = line_merge_dist  # Merge lines within this rho distance
        self.line_merge_angle = line_merge_angle  # Merge lines within this angle (degrees)
        print("[LINE_DETECTOR] Initialized with canny=({},{}), hough_threshold={}, merge_dist={}, merge_angle={}°".format(
            canny_low, canny_high, hough_threshold, line_merge_dist, line_merge_angle))
    
    def detect_lines(self, frame):
        """Main pipeline: detect grid lines"""
        print("\n" + "="*70)
        print("[STEP 1] PREPROCESSING - Grayscale and blur")
        print("="*70)
        gray = self._preprocess(frame)
        
        print("\n" + "="*70)
        print("[STEP 2] EDGE DETECTION - Canny edges")
        print("="*70)
        edges = self._detect_edges(gray)
        
        print("\n" + "="*70)
        print("[STEP 3] LINE DETECTION - Standard Hough Transform")
        print("="*70)
        lines = self._detect_hough_lines(edges)
        
        print("\n" + "="*70)
        print("[STEP 4] MERGE NEARBY LINES - Cluster similar lines")
        print("="*70)
        lines = self._merge_lines(lines)
        
        print("\n" + "="*70)
        print("[STEP 5] DRAW LINES - Visualize detected lines")
        print("="*70)
        labeled_frame = self._draw_lines(frame.copy(), lines)
        
        return labeled_frame, lines
    
    def _preprocess(self, frame):
        """Step 1: Convert to grayscale and blur"""
        print("  → Converting to grayscale...")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        print("  → Applying Gaussian blur...")
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        print(f"  ✓ Image shape: {blurred.shape}")
        return blurred
    
    def _detect_edges(self, gray):
        """Step 2: Detect edges using Canny"""
        print(f"  → Applying Canny edge detection ({self.canny_low}, {self.canny_high})...")
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        
        edge_pixels = np.count_nonzero(edges)
        print(f"  ✓ Edge pixels: {edge_pixels}")
        return edges
    
    def _detect_hough_lines(self, edges):
        """Step 3: Detect lines using Standard Hough Transform (infinite lines)"""
        print(f"  → Running Standard Hough Line Transform (infinite lines)...")
        print(f"     threshold={self.hough_threshold}")
        
        # Use standard Hough transform (returns infinite lines as (rho, theta))
        lines = cv2.HoughLines(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.hough_threshold
        )
        
        if lines is None:
            lines = []
        else:
            # Convert to list of (rho, theta) tuples
            lines = [(line[0][0], line[0][1]) for line in lines]
        
        print(f"  ✓ Detected {len(lines)} infinite lines")
        return lines
    
    def _merge_lines(self, lines):
        """Step 4: Merge nearby parallel lines to reduce duplicates"""
        if not lines:
            return []
        
        print(f"  → Starting with {len(lines)} lines")
        merged = []
        used = set()
        
        for i, (rho1, theta1) in enumerate(lines):
            if i in used:
                continue
            
            # Convert theta to degrees for comparison
            theta1_deg = theta1 * 180 / np.pi
            
            # Find all lines similar to this one
            similar = [(rho1, theta1)]
            used.add(i)
            
            for j, (rho2, theta2) in enumerate(lines):
                if j in used or i == j:
                    continue
                
                theta2_deg = theta2 * 180 / np.pi
                
                # Check if lines are parallel (similar angle)
                angle_diff = min(abs(theta1_deg - theta2_deg), 180 - abs(theta1_deg - theta2_deg))
                
                # Check if lines are close in distance
                rho_diff = abs(rho1 - rho2)
                
                # Merge if both angle and distance are similar
                if angle_diff < self.line_merge_angle and rho_diff < self.line_merge_dist:
                    similar.append((rho2, theta2))
                    used.add(j)
            
            # Average the similar lines
            avg_rho = np.mean([line[0] for line in similar])
            avg_theta = np.mean([line[1] for line in similar])
            merged.append((avg_rho, avg_theta))
        
        print(f"  ✓ After merging: {len(merged)} lines (removed {len(lines) - len(merged)} duplicates)")
        return merged
    
    def _draw_lines(self, frame, lines):
        """Step 4: Draw infinite lines across entire frame"""
        if not lines:
            print("  ✗ No lines to draw")
            return frame
        
        print(f"  → Drawing {len(lines)} lines across entire frame...")
        
        h, w = frame.shape[:2]
        
        # Classify lines as horizontal or vertical
        horizontal_lines = []
        vertical_lines = []
        
        for rho, theta in lines:
            # Line equation: x*cos(theta) + y*sin(theta) = rho
            # Calculate angle in degrees
            angle_deg = theta * 180 / np.pi
            
            # Classify based on angle (perpendicular to line direction)
            if angle_deg < 45 or angle_deg > 135:
                horizontal_lines.append((rho, theta))
            else:
                vertical_lines.append((rho, theta))
        
        # Draw horizontal lines in GREEN (extended to frame edges)
        for rho, theta in horizontal_lines:
            # Find intersection with frame boundaries
            x1, y1, x2, y2 = self._line_to_frame_edges(rho, theta, w, h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw vertical lines in BLUE (extended to frame edges)
        for rho, theta in vertical_lines:
            # Find intersection with frame boundaries
            x1, y1, x2, y2 = self._line_to_frame_edges(rho, theta, w, h)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        print(f"  ✓ Drew {len(horizontal_lines)} horizontal lines (GREEN)")
        print(f"  ✓ Drew {len(vertical_lines)} vertical lines (BLUE)")
        
        return frame
    
    def _line_to_frame_edges(self, rho, theta, w, h):
        """Calculate where a line intersects with frame boundaries"""
        a = np.cos(theta)
        b = np.sin(theta)
        
        points = []
        
        # Left edge (x=0): y = rho / sin(theta)
        if abs(b) > 1e-6:
            y = rho / b
            if 0 <= y <= h:
                points.append((0, int(y)))
        
        # Right edge (x=w): y = (rho - w*cos(theta)) / sin(theta)
        if abs(b) > 1e-6:
            y = (rho - w * a) / b
            if 0 <= y <= h:
                points.append((w, int(y)))
        
        # Top edge (y=0): x = rho / cos(theta)
        if abs(a) > 1e-6:
            x = rho / a
            if 0 <= x <= w:
                points.append((int(x), 0))
        
        # Bottom edge (y=h): x = (rho - h*sin(theta)) / cos(theta)
        if abs(a) > 1e-6:
            x = (rho - h * b) / a
            if 0 <= x <= w:
                points.append((int(x), h))
        
        # Return two edge points (or duplicate if only one found)
        if len(points) >= 2:
            return points[0][0], points[0][1], points[1][0], points[1][1]
        elif len(points) == 1:
            return points[0][0], points[0][1], points[0][0], points[0][1]
        else:
            # Fallback: extend in both directions
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + w * (-b))
            y1 = int(y0 + w * a)
            x2 = int(x0 - w * (-b))
            y2 = int(y0 - w * a)
            return x1, y1, x2, y2

