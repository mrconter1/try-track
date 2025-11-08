import cv2
import numpy as np
import time

class LineDetector:
    def __init__(self, canny_low=30, canny_high=100, hough_threshold=150, line_merge_dist=30, line_merge_angle=5):
        """Initialize line detector for grid lines"""
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.line_merge_dist = line_merge_dist  # Merge lines within this rho distance
        self.line_merge_angle = line_merge_angle  # Merge lines within this angle (degrees)
    
    def detect_lines(self, frame):
        """Main pipeline: detect grid lines"""
        start_time = time.time()
        
        gray = self._preprocess(frame)
        edges = self._detect_edges(gray)
        lines = self._detect_hough_lines(edges)
        lines = self._merge_lines(lines)
        labeled_frame = self._draw_lines(frame.copy(), lines)
        
        elapsed_ms = (time.time() - start_time) * 1000
        print(f"[LINE_DETECTOR] Frame processed in {elapsed_ms:.2f}ms | Lines detected: {len(lines)}")
        
        return labeled_frame, lines
    
    def _preprocess(self, frame):
        """Step 1: Convert to grayscale and blur"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred
    
    def _detect_edges(self, gray):
        """Step 2: Detect edges using Canny"""
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        return edges
    
    def _detect_hough_lines(self, edges):
        """Step 3: Detect lines using Standard Hough Transform (infinite lines)"""
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
        
        return lines
    
    def _merge_lines(self, lines):
        """Step 4: Merge nearby parallel lines to reduce duplicates"""
        if not lines:
            return []
        
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
        
        return merged
    
    def _draw_lines(self, frame, lines):
        """Draw infinite lines across entire frame"""
        if not lines:
            return frame
        
        h, w = frame.shape[:2]
        
        # Draw all lines in BLUE (extended to frame edges)
        for rho, theta in lines:
            # Find intersection with frame boundaries
            x1, y1, x2, y2 = self._line_to_frame_edges(rho, theta, w, h)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
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

