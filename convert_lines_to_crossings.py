"""
Convert line_annotations.json to cross_annotations.json

Computes intersection points between line segments in each sample
and saves them as crossing annotations.

Usage: python convert_lines_to_crossings.py
       python convert_lines_to_crossings.py --input line_annotations.json --output cross_annotations.json
"""

import json
import argparse
import os
from typing import List, Tuple


def find_line_intersections(lines: List[dict]) -> List[Tuple[float, float]]:
    """
    Find all intersection points between line segments.
    
    Args:
        lines: List of line dicts with 'start' and 'end' keys, each being [x, y]
    
    Returns:
        List of (x, y) intersection coordinates
    """
    intersections = []
    
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            line1 = lines[i]
            line2 = lines[j]
            
            # Line 1: from (x1, y1) to (x2, y2)
            x1, y1 = line1['start']
            x2, y2 = line1['end']
            
            # Line 2: from (x3, y3) to (x4, y4)
            x3, y3 = line2['start']
            x4, y4 = line2['end']
            
            # Calculate intersection using parametric form
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            
            if abs(denom) < 1e-10:
                continue  # Lines are parallel
            
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
            
            # Check if intersection is within both line segments
            if 0 <= t <= 1 and 0 <= u <= 1:
                ix = x1 + t * (x2 - x1)
                iy = y1 + t * (y2 - y1)
                intersections.append((ix, iy))
    
    return intersections


def convert_annotations(input_file: str, output_file: str):
    """Convert line annotations to crossing annotations."""
    
    # Load input
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        return False
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    samples = data.get('samples', [])
    print(f"Loaded {len(samples)} samples from {input_file}")
    
    # Process each sample
    cross_samples = []
    total_crossings = 0
    samples_with_crossings = 0
    
    for sample in samples:
        lines = sample.get('lines', [])
        
        # Find intersections
        crossings = find_line_intersections(lines)
        
        # Create crossing annotation
        cross_sample = {
            'video_path': sample['video_path'],
            'frame_idx': sample['frame_idx'],
            'crop_rect': sample['crop_rect'],
            'crossings': [[x, y] for x, y in crossings],
            'num_lines': len(lines)
        }
        
        cross_samples.append(cross_sample)
        
        if crossings:
            total_crossings += len(crossings)
            samples_with_crossings += 1
    
    # Save output
    output_data = {
        'samples': cross_samples,
        'stats': {
            'total_samples': len(cross_samples),
            'samples_with_crossings': samples_with_crossings,
            'total_crossings': total_crossings
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nConversion complete!")
    print(f"  Total samples: {len(cross_samples)}")
    print(f"  Samples with crossings: {samples_with_crossings}")
    print(f"  Total crossings: {total_crossings}")
    print(f"  Average crossings per sample: {total_crossings / len(samples):.2f}")
    print(f"\nSaved to: {output_file}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert line annotations to crossing annotations")
    parser.add_argument("--input", "-i", default="line_annotations.json", help="Input line annotations file")
    parser.add_argument("--output", "-o", default="cross_annotations.json", help="Output crossing annotations file")
    
    args = parser.parse_args()
    
    convert_annotations(args.input, args.output)


if __name__ == "__main__":
    main()

