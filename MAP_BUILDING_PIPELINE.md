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
      
      // Store observation and triangulate if seen from multiple cameras
      register_tile_observation(tile_id, square.position, frame.camera_id)
    
    END FOR
    
    // Periodically optimize all tile positions using bundle adjustment
    IF frame_count % OPTIMIZATION_INTERVAL == 0:
      optimize_global_map(map_tiles, all_observations)
    END IF
  
  END FOR

RETURN global_map = build_map_output(map_tiles, map_coordinates)
```
