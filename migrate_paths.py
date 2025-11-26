#!/usr/bin/env python3
"""
Migration script to convert absolute video paths to relative paths in line_annotations.json.
This makes the annotation database portable across machines.

Usage:
    python migrate_paths.py
    python migrate_paths.py --dry-run  # Preview changes without saving
"""

import json
import os
import argparse
import shutil
from datetime import datetime


def get_basename(path):
    """Extract filename from path, handling both Windows and Unix separators."""
    return path.replace('\\', '/').split('/')[-1]


def migrate_paths(db_path="line_annotations.json", video_folder="videos", dry_run=False):
    """Convert absolute paths to relative paths."""
    
    if not os.path.exists(db_path):
        print(f"[ERROR] Database not found: {db_path}")
        return False
    
    # Load database
    print(f"[INFO] Loading {db_path}...")
    with open(db_path, 'r') as f:
        data = json.load(f)
    
    samples = data.get('samples', [])
    print(f"[INFO] Found {len(samples)} samples")
    
    # Track changes
    changes = 0
    not_found = []
    
    for sample in samples:
        old_path = sample.get('video_path', '')
        basename = get_basename(old_path)
        
        # Create relative path
        new_path = f"{video_folder}/{basename}"
        
        # Check if file exists
        if not os.path.exists(new_path):
            if basename not in [get_basename(p) for p in not_found]:
                not_found.append(old_path)
        
        if old_path != new_path:
            if dry_run:
                print(f"  [CHANGE] {old_path}")
                print(f"        -> {new_path}")
            sample['video_path'] = new_path
            changes += 1
    
    print(f"\n[INFO] Changes: {changes} paths updated")
    
    if not_found:
        print(f"\n[WARN] {len(not_found)} videos not found in '{video_folder}/':")
        for path in not_found[:10]:  # Show first 10
            print(f"  - {get_basename(path)}")
        if len(not_found) > 10:
            print(f"  ... and {len(not_found) - 10} more")
    
    if dry_run:
        print(f"\n[DRY RUN] No changes saved. Remove --dry-run to apply changes.")
        return True
    
    if changes == 0:
        print("[INFO] No changes needed.")
        return True
    
    # Backup original
    backup_path = f"line_annotations_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    print(f"\n[INFO] Creating backup: {backup_path}")
    shutil.copy(db_path, backup_path)
    
    # Save updated database
    print(f"[INFO] Saving updated {db_path}...")
    with open(db_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n[DONE] Successfully migrated {changes} paths to relative format!")
    print(f"[INFO] Backup saved to: {backup_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Migrate absolute paths to relative paths")
    parser.add_argument("--db", type=str, default="line_annotations.json", 
                        help="Path to annotation database (default: line_annotations.json)")
    parser.add_argument("--video-folder", type=str, default="videos",
                        help="Video folder name (default: videos)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without saving")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PATH MIGRATION SCRIPT")
    print("=" * 60)
    print(f"Database: {args.db}")
    print(f"Video folder: {args.video_folder}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 60)
    
    migrate_paths(args.db, args.video_folder, args.dry_run)


if __name__ == "__main__":
    main()

