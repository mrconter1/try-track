# Tile-Based Map Building Pipeline

```
GLOBAL_STATE:
  map_tiles = {}                    // Store tile images and metadata
  tile_signatures = {}              // tile_id -> hash signature
  map_coordinates = {}              // tile_id -> 3D world position
  next_tile_id = 0

MAIN LOOP:
  FOR EACH frame IN video_stream:
    
    // Detect lines: grayscale -> edge detection -> Hough transform
    line_segments = detect_lines(frame)
    
    // Filter false lines by angle clustering and spacing consistency
    filtered_lines = filter_false_lines(line_segments)
    
    // Find grid intersections from horizontal/vertical lines
    grid_squares = create_grid_from_lines(filtered_lines, frame.shape)
    
    // Extract all tile images and apply perspective warp to 1:1 ratio (perfect squares)
    tile_images = extract_tile_images(grid_squares, frame)
    
    // Filter to keep only uniform tiles (remove ones with feet/shadows)
    uniform_tile_images = filter_tiles_by_quality(tile_images, MAX_STD_THRESHOLD)
    
    FOR EACH tile_image IN uniform_tile_images:
      
      // Generate hash signature (16x16 blocks, CLAHE preprocessing)
      tile_signature = create_tile_signature(tile_image)
      
      // Compare against existing tile database
      match = find_closest_match(tile_signature, tile_signatures)
      
      // Either link to existing tile or create new one
      IF match exists AND confidence > THRESHOLD:
        tile_id = match.tile_id
      ELSE:
        tile_id = next_tile_id++
        tile_signatures[tile_id] = tile_signature
        map_tiles[tile_id] = new_tile_entry(tile_id, tile_image)
      END IF
    
    END FOR
  
  END FOR

RETURN global_map = build_map_output(map_tiles, map_coordinates)
```

---

# Global Map Building (Post-Processing)

```
INPUT:
  grids = [
    {
      frame_number,
      grid: 2D array where each cell is tile_signature (or empty)
    },
    ...  // One grid per frame
  ]

GLOBAL_MAP:
  tile_global_positions = {}        // signature -> global_coords
  frame_offsets = {}                // frame_number -> offset_transform

BUILD_GLOBAL_MAP:
  FOR EACH grid_frame IN grids:
    // Find first tile signature in grid that matches existing global map
    first_match = find_first_matching_tile_in_grid(grid_frame.grid, tile_global_positions)
    
    IF first_match != NULL:
      (grid_row, grid_col, signature) = first_match
      offset = tile_global_positions[signature] - (grid_row, grid_col)
      
      // Stitch entire grid into global map using offset
      FOR EACH (row, col) IN grid_frame.grid:
        IF grid_frame.grid[row][col] != empty:
          global_coords = (row, col) + offset
          tile_global_positions[grid_frame.grid[row][col]] = global_coords
        END IF
      END FOR
    END IF
  END FOR

RETURN global_map_with_coordinates
```
