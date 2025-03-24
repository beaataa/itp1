#!/usr/bin/env python3
"""
This script identifies image files with the same filename across three folders
and copies those duplicates to a new folder.

Place this script in the directory containing your image folders, then run:
python duplicate_finder.py
"""

import os
import shutil
from collections import defaultdict


def find_duplicate_filenames(folder_paths):
    """
    Find filenames that appear more than once across the provided folders.
    
    Args:
        folder_paths: List of folder paths to search
        
    Returns:
        Dictionary mapping filenames to lists of their full paths
    """
    # Dictionary to store filename -> list of full paths
    file_paths = defaultdict(list)
    
    # Scan all folders
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist. Skipping.")
            continue
            
        for filename in os.listdir(folder_path):
            # Check if the file is an image (basic check based on extension)
            if is_image_file(filename):
                full_path = os.path.join(folder_path, filename)
                file_paths[filename].append(full_path)
    
    # Filter to keep only filenames that appear more than once
    duplicate_files = {
        filename: paths for filename, paths in file_paths.items() 
        if len(paths) > 1
    }
    
    return duplicate_files


def is_image_file(filename):
    """Check if a file is an image based on its extension."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    _, ext = os.path.splitext(filename.lower())
    return ext in image_extensions


def copy_duplicates_to_destination(duplicate_files, destination_folder):
    """
    Copy all identified duplicate files to the destination folder.
    
    Args:
        duplicate_files: Dictionary mapping filenames to lists of their full paths
        destination_folder: Path to destination folder
    """
    # Create destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Created destination folder: {destination_folder}")
    
    # Copy each duplicate file
    for filename, paths in duplicate_files.items():
        print(f"Found duplicate: {filename} in {len(paths)} locations")
        
        # Copy each instance of the duplicate
        for i, src_path in enumerate(paths):
            if i == 0:
                # First instance keeps the original filename
                dst_path = os.path.join(destination_folder, filename)
            else:
                # For additional instances, add a suffix to avoid overwriting
                base, ext = os.path.splitext(filename)
                dst_path = os.path.join(destination_folder, f"{base}_dup{i}{ext}")
            
            shutil.copy2(src_path, dst_path)
            print(f"  - Copied from {src_path} to {dst_path}")


def main():
    current_dir = os.getcwd()
    
    # Find all directories in the current folder (excluding the duplicates folder)
    all_dirs = [d for d in os.listdir(current_dir) 
               if os.path.isdir(os.path.join(current_dir, d)) and d != "duplicates"]
    
    if len(all_dirs) < 3:
        print(f"Warning: Found only {len(all_dirs)} folders in the current directory.")
        print("This script is designed to find duplicates across at least 3 folders.")
        print(f"Folders found: {', '.join(all_dirs)}")
        
        # Ask for confirmation
        response = input("Do you want to continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            print("Exiting script.")
            return
    
    # Make full paths for the source folders
    folder_paths = [os.path.join(current_dir, folder) for folder in all_dirs]
    
    # Set destination folder
    destination_folder = os.path.join(current_dir, "duplicates")
    
    print(f"Searching for duplicate filenames across {len(folder_paths)} folders:")
    for folder in folder_paths:
        print(f"  - {folder}")
    print(f"Duplicates will be copied to: {destination_folder}")
    
    # Find duplicate filenames
    duplicate_files = find_duplicate_filenames(folder_paths)
    
    if not duplicate_files:
        print("No duplicate filenames found across the provided folders.")
        return
    
    # Copy duplicate files to destination
    copy_duplicates_to_destination(duplicate_files, destination_folder)
    
    # Print summary
    total_duplicates = sum(len(paths) for paths in duplicate_files.values())
    print(f"\nSummary: Found {len(duplicate_files)} unique filenames with duplicates")
    print(f"Total duplicate files copied: {total_duplicates}")


if __name__ == "__main__":
    main()