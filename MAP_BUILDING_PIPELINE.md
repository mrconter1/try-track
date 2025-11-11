# Tile-Based Map Building Pipeline

## Data Classes

```
CLASS TileSignature:
  signature_0: string
  signature_90: string
  signature_180: string
  signature_270: string

CLASS Tile:
  tile_id: int
  signatures: TileSignature
  tile_image: image

CLASS Frame:
  frame_number: int
  grid: 2D array of (tile_id or empty)

CLASS GlobalMap:
  tile_global_positions: dict[tile_id -> (x, y)]
```

## Frame Processing Pipeline

```
GLOBAL_STATE:
  tiles = []                        // Array of Tile objects
  frames = []                       // Array of Frame objects
  next_tile_id = 0

MAIN LOOP:
  FOR EACH video_frame IN video_stream:
    
    // Detect lines: grayscale -> edge detection -> Hough transform
    line_segments = detect_lines(video_frame)
    
    // Filter false lines by angle clustering and spacing consistency
    filtered_lines = filter_false_lines(line_segments)
    
    // Find grid intersections from horizontal/vertical lines
    grid_squares = create_grid_from_lines(filtered_lines, video_frame.shape)
    
    // Extract all tile images and apply perspective warp to 1:1 ratio (perfect squares)
    tile_images = extract_tile_images(grid_squares, video_frame)
    
    // Filter to keep only uniform tiles (remove ones with feet/shadows)
    uniform_tile_images = filter_tiles_by_quality(tile_images, MAX_STD_THRESHOLD)
    
    frame_grid = create_empty_grid(tile_images.dimensions)
    
    FOR EACH (row, col, tile_image) IN uniform_tile_images:
      
      // Generate hash signatures for all 4 rotations (16x16 blocks, CLAHE preprocessing)
      signatures = new TileSignature()
      FOR EACH rotation IN [0, 90, 180, 270]:
        rotated_image = rotate_image(tile_image, rotation)
        signatures[rotation] = create_tile_signature(rotated_image)
      END FOR
      
      // Create and store tile
      tile = new Tile(next_tile_id++, signatures, tile_image)
      tiles.append(tile)
      frame_grid[row][col] = tile.tile_id
    
    END FOR
    
    frame = new Frame(video_frame.frame_number, frame_grid)
    frames.append(frame)
  
  END FOR

RETURN frames
```

---

# Global Map Building (Post-Processing)

```
INPUT:
  frames: Frame[]           // Array of Frame objects from pipeline

OUTPUT:
  global_map: GlobalMap     // Tile positions in global coordinates

BUILD_GLOBAL_MAP:
  tile_global_positions = {}        // tile_id -> global coordinates
  
  FOR EACH frame IN frames:
    // Find first tile in grid that matches existing global map
    first_match = find_first_matching_tile_in_grid(frame.grid, tile_global_positions)
    
    IF first_match != NULL:
      (grid_row, grid_col, tile_id) = first_match
      offset = tile_global_positions[tile_id] - (grid_row, grid_col)
      
      // Stitch entire grid into global map using offset
      FOR EACH (row, col) IN frame.grid:
        IF frame.grid[row][col] != empty:
          global_coords = (row, col) + offset
          tile_global_positions[frame.grid[row][col]] = global_coords
        END IF
      END FOR
    END IF
  END FOR

  global_map = new GlobalMap(tile_global_positions)
  RETURN global_map
```
