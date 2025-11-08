import cv2
import numpy as np

class ContourDetector:
    def __init__(self, min_area=1000, max_area=50000, epsilon=0.05, side_ratio_tolerance=0.08, angle_tolerance=10):
        """Initialize contour detector for enclosed squares"""
        self.min_area = min_area
        self.max_area = max_area
        self.epsilon = epsilon
        self.side_ratio_tolerance = side_ratio_tolerance
        self.angle_tolerance = angle_tolerance
        self.prev_frame_tiles = {}  # Store tiles from previous frame
        self.frame_counter = 0
        
        # Global unique tile tracking
        self.unique_tiles = {}  # tile_id -> {seen_count, hashes, positions[], rotations[], last_row, last_col}
        self.next_tile_id = 0
        
        print("[CONTOUR_DETECTOR] Initialized with min_area={}, max_area={}, epsilon={}, side_ratio_tolerance={}, angle_tolerance={}°".format(
            min_area, max_area, epsilon, side_ratio_tolerance, angle_tolerance))
    
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
        edges = cv2.Canny(gray, 10, 50)
        
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
        """Step 5: Filter contours to keep only 4-sided shapes that are squares"""
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
                # Check if it's actually a square
                if self._is_square(approx):
                    quadrilaterals.append(approx)
        
        print(f"  ✓ Kept {len(quadrilaterals)} valid squares")
        return quadrilaterals
    
    def _is_square(self, quad):
        """Check if quadrilateral is approximately a square"""
        # Get the 4 vertices
        pts = quad.reshape(4, 2).astype(float)
        
        # ===== CHECK 1: All 4 sides should be equal length =====
        side_lengths = []
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            side_lengths.append(dist)
        
        avg_side = np.mean(side_lengths)
        if avg_side == 0:
            return False
        
        # All sides should be within tolerance of average
        for side in side_lengths:
            if abs(side - avg_side) / avg_side > self.side_ratio_tolerance:
                return False
        
        # ===== CHECK 2: Both diagonals should be equal length =====
        diag1 = np.sqrt((pts[0][0] - pts[2][0])**2 + (pts[0][1] - pts[2][1])**2)
        diag2 = np.sqrt((pts[1][0] - pts[3][0])**2 + (pts[1][1] - pts[3][1])**2)
        
        avg_diag = (diag1 + diag2) / 2
        if avg_diag == 0:
            return False
        
        # Diagonals should be approximately equal
        if abs(diag1 - diag2) / avg_diag > self.side_ratio_tolerance:
            return False
        
        # ===== CHECK 3: Diagonals should relate to sides correctly =====
        # In a square: diagonal = side * sqrt(2) ≈ side * 1.414
        expected_diag = avg_side * np.sqrt(2)
        if expected_diag == 0:
            return False
        
        diag_ratio = avg_diag / expected_diag
        if abs(diag_ratio - 1.0) > self.side_ratio_tolerance:
            return False
        
        # ===== CHECK 4: All 4 angles should be ~90 degrees =====
        angles = []
        for i in range(4):
            # Get three consecutive points to form angle
            p1 = pts[(i - 1) % 4]
            p2 = pts[i]
            p3 = pts[(i + 1) % 4]
            
            # Vectors from p2 to p1 and p2 to p3
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Calculate angle using dot product
            dot_product = np.dot(v1, v2)
            cross_product = np.abs(v1[0] * v2[1] - v1[1] * v2[0])
            
            angle_rad = np.arctan2(cross_product, dot_product)
            angle_deg = np.degrees(angle_rad)
            
            angles.append(angle_deg)
        
        # All angles should be close to 90 degrees
        for angle in angles:
            if abs(angle - 90) > self.angle_tolerance:
                return False
        
        # ===== CHECK 5: Aspect ratio should be reasonable =====
        x, y, w, h = cv2.boundingRect(quad)
        aspect_ratio = float(w) / h if h != 0 else 0
        if not (0.5 < aspect_ratio < 2.0):
            return False
        
        return True
    
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
    
    def extract_and_correct_tiles(self, frame, grid_tiles, tile_size=150):
        """Extract and perspective-correct each tile, compute hashes for all rotations, store images"""
        corrected_tiles = {}
        tile_hashes = {}
        self.current_frame_images = {}  # Store current frame images for matching
        
        for (row, col), quad in grid_tiles.items():
            # Get the 4 corner points
            pts = quad.reshape(4, 2).astype(np.float32)
            
            # Reorder points to ensure consistent ordering: top-left, top-right, bottom-right, bottom-left
            # Calculate center
            center = pts.mean(axis=0)
            
            # Sort points by angle from center for consistent ordering
            angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
            sorted_indices = np.argsort(angles)
            pts_sorted = pts[sorted_indices]
            
            # Reorder to: top-left, top-right, bottom-right, bottom-left
            # Find the top point (minimum y)
            top_idx = np.argmin(pts_sorted[:, 1])
            pts_ordered = np.roll(pts_sorted, -top_idx, axis=0)
            
            # Define destination points (perfect square)
            dst_pts = np.array([
                [0, 0],
                [tile_size, 0],
                [tile_size, tile_size],
                [0, tile_size]
            ], dtype=np.float32)
            
            try:
                # Get perspective transformation matrix
                M = cv2.getPerspectiveTransform(pts_ordered, dst_pts)
                
                # Apply perspective warp to extract and straighten the tile
                corrected = cv2.warpPerspective(frame, M, (tile_size, tile_size))
                
                corrected_tiles[(row, col)] = corrected
                self.current_frame_images[(row, col)] = corrected  # Store for unique tile database
                
                # Normalize tile for better hash consistency
                normalized = self._normalize_tile(corrected)
                
                # Compute hashes for all 4 rotations
                hashes = {}
                for rotation in [0, 90, 180, 270]:
                    rotated = self._rotate_image(normalized, rotation)
                    hashes[rotation] = self._compute_dhash(rotated)
                
                tile_hashes[(row, col)] = hashes
                
            except cv2.error as e:
                print(f"  ✗ Error perspective correcting tile ({row},{col}): {e}")
        
        return corrected_tiles, tile_hashes
    
    def _normalize_tile(self, tile_image):
        """Normalize tile image for consistent hashing across different lighting"""
        # Convert to grayscale if needed
        if len(tile_image.shape) == 3:
            gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = tile_image
        
        # Histogram equalization for lighting invariance
        normalized = cv2.equalizeHist(gray)
        return normalized
    
    def _rotate_image(self, image, angle):
        """Rotate image by angle (0, 90, 180, 270)"""
        if angle == 0:
            return image
        elif angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return image
    
    def _compute_dhash(self, image, hash_size=8):
        """Compute difference hash (dHash) of an image"""
        # Resize image to hash_size x (hash_size+1)
        resized = cv2.resize(image, (hash_size + 1, hash_size))
        
        # Compute differences between adjacent pixels
        diff = resized[:, 1:] > resized[:, :-1]
        
        # Convert boolean array to hex string
        hash_bytes = 0
        for i, row in enumerate(diff):
            for j, val in enumerate(row):
                if val:
                    hash_bytes += (1 << (i * hash_size + j))
        
        # Return as hex string
        return format(hash_bytes, f'0{hash_size * hash_size // 4}x')
    
    def _hamming_distance(self, hash1, hash2):
        """Compute Hamming distance between two hex hash strings"""
        if not hash1 or not hash2:
            return float('inf')
        
        try:
            # Convert hex to integers
            val1 = int(hash1, 16)
            val2 = int(hash2, 16)
            # XOR to find differing bits
            xor = val1 ^ val2
            # Count 1s in binary representation (Hamming distance)
            return bin(xor).count('1')
        except:
            return float('inf')
    
    def match_tiles_to_previous(self, current_tiles_hashes):
        """Match tiles from current frame to unique tile database - rotation doesn't matter"""
        matches = {}  # Maps current tile pos to unique_tile_id
        
        # If no previous frame, just store current and return empty matches
        if not self.prev_frame_tiles:
            print(f"  → First frame, storing {len(current_tiles_hashes)} tiles for next frame")
            self.prev_frame_tiles = current_tiles_hashes
            return matches
        
        print(f"  → Matching current {len(current_tiles_hashes)} tiles to {len(self.unique_tiles)} unique tiles...")
        
        # For each tile in current frame
        for curr_pos, curr_hashes in current_tiles_hashes.items():
            best_tile_id = None
            best_distance = float('inf')
            best_curr_rotation = None
            
            # Try to match against all known unique tiles
            # Check if ANY rotation of current tile matches ANY rotation of any known tile
            for tile_id, tile_data in self.unique_tiles.items():
                known_hashes = tile_data['hashes']
                
                # Find best rotation match for this tile
                for curr_rot in [0, 90, 180, 270]:
                    for known_rot in [0, 90, 180, 270]:
                        curr_hash = curr_hashes.get(curr_rot, "")
                        known_hash = known_hashes.get(known_rot, "")
                        
                        distance = self._hamming_distance(curr_hash, known_hash)
                        
                        # Keep track of best match (lowest distance across all tiles and rotations)
                        if distance < best_distance:
                            best_distance = distance
                            best_tile_id = tile_id
                            best_curr_rotation = curr_rot
            
            # If good match found to unique tile, count as same sighting
            if best_distance < 20 and best_tile_id is not None:
                matches[curr_pos] = best_tile_id
                self.unique_tiles[best_tile_id]['seen_count'] += 1
                self.unique_tiles[best_tile_id]['last_row'] = curr_pos[0]
                self.unique_tiles[best_tile_id]['last_col'] = curr_pos[1]
                self.unique_tiles[best_tile_id]['last_rotation'] = best_curr_rotation
            else:
                # No good match - this is a new tile
                matches[curr_pos] = None
        
        # Add new unique tiles (those with None matches)
        for curr_pos, tile_id in list(matches.items()):
            if tile_id is None:
                # Create new unique tile
                new_id = self.next_tile_id
                self.next_tile_id += 1
                
                # Get the corrected image for this tile if available
                tile_image = self.current_frame_images.get(curr_pos, None)
                
                self.unique_tiles[new_id] = {
                    'seen_count': 1,
                    'hashes': current_tiles_hashes[curr_pos],  # Store all 4 rotation hashes
                    'image': tile_image,
                    'first_frame': self.frame_counter,
                    'last_row': curr_pos[0],
                    'last_col': curr_pos[1],
                    'last_rotation': 0,
                    'positions': [curr_pos]
                }
                matches[curr_pos] = new_id
                print(f"  → New unique tile #{new_id} at {curr_pos}")
        
        # Store current frame for next iteration
        self.prev_frame_tiles = current_tiles_hashes
        self.frame_counter += 1
        
        matched_count = len([m for m in matches.values() if m is not None])
        new_count = len([m for m in matches.values() if m is None])
        print(f"  ✓ Matched {matched_count} tiles | New tiles: {new_count} | Total unique: {len(self.unique_tiles)}")
        return matches

