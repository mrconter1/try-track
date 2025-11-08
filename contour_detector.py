import cv2
import numpy as np

class ContourDetector:
    def __init__(self, min_area=1000, max_area=50000, epsilon=0.05):
        """Initialize contour detector for enclosed squares"""
        self.min_area = min_area
        self.max_area = max_area
        self.epsilon = epsilon
        print("[CONTOUR_DETECTOR] Initialized with min_area={}, max_area={}, epsilon={}".format(
            min_area, max_area, epsilon))
    
    def detect_tiles(self, frame):
        """Main pipeline: detect enclosed skewed squares"""
        print("\n" + "="*70)
        print("[STEP 1] PREPROCESSING - Grayscale and blur")
        print("="*70)
        gray = self._preprocess(frame)
        
        print("\n" + "="*70)
        print("[STEP 2] EDGE DETECTION - Canny edges")
        print("="*70)
        edges = self._detect_edges(gray)
        
        print("\n" + "="*70)
        print("[STEP 3] MORPHOLOGY - Enhance boundaries")
        print("="*70)
        enhanced = self._enhance_edges(edges)
        
        print("\n" + "="*70)
        print("[STEP 4] CONTOUR DETECTION - Find all enclosed shapes")
        print("="*70)
        contours = self._find_contours(enhanced)
        
        print("\n" + "="*70)
        print("[STEP 5] FILTER QUADRILATERALS - Keep 4-sided shapes")
        print("="*70)
        quadrilaterals = self._filter_quadrilaterals(contours)
        
        print("\n" + "="*70)
        print("[STEP 6] ORGANIZE GRID - Sort tiles by position")
        print("="*70)
        grid_tiles = self._organize_grid(quadrilaterals)
        
        print("\n" + "="*70)
        print("[STEP 7] DRAW & LABEL - Visualize tiles")
        print("="*70)
        labeled_frame = self._draw_and_label_tiles(frame.copy(), grid_tiles)
        
        return labeled_frame, grid_tiles
    
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
        print("  → Applying Canny edge detection...")
        edges = cv2.Canny(gray, 50, 150)
        
        edge_pixels = np.count_nonzero(edges)
        print(f"  ✓ Edge pixels: {edge_pixels}")
        return edges
    
    def _enhance_edges(self, edges):
        """Step 3: Enhance edges using morphological operations"""
        print("  → Closing small holes in edges...")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        print("  → Dilating edges...")
        dilated = cv2.dilate(closed, kernel, iterations=1)
        
        print("  ✓ Edges enhanced")
        return dilated
    
    def _find_contours(self, enhanced):
        """Step 4: Find all contours in enhanced edge image"""
        print("  → Finding all contours...")
        contours, _ = cv2.findContours(enhanced, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"  ✓ Found {len(contours)} contours")
        return contours
    
    def _filter_quadrilaterals(self, contours):
        """Step 5: Filter contours to keep only 4-sided shapes"""
        print(f"  → Filtering {len(contours)} contours...")
        quadrilaterals = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Check area
            if area < self.min_area or area > self.max_area:
                continue
            
            # Approximate contour to polygon
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, self.epsilon * perimeter, True)
            
            # Check if it's a quadrilateral (4 vertices)
            if len(approx) == 4:
                # Calculate aspect ratio to filter out very elongated shapes
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h != 0 else 0
                
                # Keep shapes that are roughly square-ish (0.5 to 2.0 aspect ratio)
                if 0.5 < aspect_ratio < 2.0:
                    quadrilaterals.append(approx)
        
        print(f"  ✓ Kept {len(quadrilaterals)} quadrilaterals with good aspect ratio")
        return quadrilaterals
    
    def _organize_grid(self, quadrilaterals):
        """Step 6: Organize tiles into grid structure by position"""
        if not quadrilaterals:
            print("  ✗ No quadrilaterals to organize")
            return {}
        
        print(f"  → Organizing {len(quadrilaterals)} tiles...")
        
        # Get center of each tile
        centers = []
        for quad in quadrilaterals:
            M = cv2.moments(quad)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy, quad))
        
        # Sort by y (top to bottom), then x (left to right)
        centers_sorted = sorted(centers, key=lambda p: (p[1], p[0]))
        
        # Group into rows based on y-coordinate similarity
        rows = []
        current_row = []
        row_threshold = 30
        
        for cx, cy, quad in centers_sorted:
            if not current_row:
                current_row.append((cx, cy, quad))
            else:
                # Check if this tile is in the same row as previous
                if abs(cy - current_row[0][1]) < row_threshold:
                    current_row.append((cx, cy, quad))
                else:
                    # New row
                    rows.append(current_row)
                    current_row = [(cx, cy, quad)]
        
        if current_row:
            rows.append(current_row)
        
        # Create grid dictionary
        grid = {}
        for row_idx, row in enumerate(rows):
            # Sort within row by x-coordinate
            row_sorted = sorted(row, key=lambda p: p[0])
            for col_idx, (cx, cy, quad) in enumerate(row_sorted):
                grid[(row_idx, col_idx)] = quad
        
        print(f"  ✓ Organized into approximately {len(rows)} rows and {max(len(r) for r in rows) if rows else 0} columns")
        return grid
    
    def _draw_and_label_tiles(self, frame, grid_tiles):
        """Step 7: Draw and label all detected tiles"""
        if not grid_tiles:
            print("  ✗ No tiles to draw")
            return frame
        
        print(f"  → Drawing {len(grid_tiles)} tiles...")
        
        for (row, col), quad in grid_tiles.items():
            # Draw the quadrilateral
            cv2.drawContours(frame, [quad], 0, (0, 255, 0), 2)
            
            # Calculate center
            M = cv2.moments(quad)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Draw label
                label = f"({row},{col})"
                cv2.putText(frame, label, (cx - 20, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        print(f"  ✓ Drew {len(grid_tiles)} labeled tiles")
        return frame

