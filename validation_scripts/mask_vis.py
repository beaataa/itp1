#!/usr/bin/env python3
"""
Simple Semantic Mask Visualizer

This script creates colored visualizations of semantic mask files.
Just run it in the directory containing your semantic masks.

Usage:
  python simple_mask_visualizer.py --mask-dir <mask_directory> --output-dir <output_directory>
"""

import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Visualize semantic masks with colors')
    parser.add_argument('--mask-dir', type=str, default='.', help='Directory containing the semantic masks')
    parser.add_argument('--output-dir', type=str, default='mask_visualizations', help='Directory to save visualizations')
    parser.add_argument('--class-id', type=int, default=None, help='Specific class ID to highlight (default: all classes)')
    parser.add_argument('--file-pattern', type=str, default='*_semantic_mask.png', help='Pattern to match mask files')
    return parser.parse_args()

def create_color_map():
    """Create a color map for visualizing different semantic classes"""
    # Define a color map for different classes (RGB format for matplotlib)
    colors = {
        0: [0, 0, 0],      # Background: black
        1: [0, 0, 255],    # Class 1: red
        2: [0, 255, 0],    # Class 2: green
        3: [255, 0, 0],    # Class 3: blue
        4: [255, 0, 0],    # Class 4 (Cable): blue
        5: [255, 0, 255],  # Class 5 (Connector): magenta
        6: [0, 255, 255],  # Class 6: yellow
        7: [255, 255, 0],  # Class 7: cyan
        8: [0, 0, 128],    # Class 8: dark red
        9: [0, 128, 0],    # Class 9: dark green
        10: [128, 0, 0],   # Class 10: dark blue
    }
    return colors

def visualize_mask(mask_file, output_dir, class_id=None):
    """Create and save a colored visualization of a semantic mask"""
    # Load mask
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error loading mask: {mask_file}")
        return False
    
    # Create a colored visualization
    height, width = mask.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Get color map
    colors = create_color_map()
    
    # Apply colors to mask
    if class_id is not None:
        # Only highlight specific class
        colored_mask[mask == class_id] = colors.get(class_id, [255, 0, 0])
    else:
        # Visualize all classes
        unique_classes = np.unique(mask)
        for class_val in unique_classes:
            colored_mask[mask == class_val] = colors.get(class_val, [0, 0, 0])
    
    # Create output filename
    mask_basename = os.path.basename(mask_file)
    output_basename = f"colored_{mask_basename}"
    output_path = os.path.join(output_dir, output_basename)
    
    # Get unique class values and their counts
    unique_classes = np.unique(mask)
    class_counts = {cls: np.sum(mask == cls) for cls in unique_classes}
    
    # Create class labels for the figure
    class_labels = []
    for cls in unique_classes:
        pixel_count = class_counts[cls]
        percentage = (pixel_count / (height * width)) * 100
        
        class_name = f"Class {cls}"
        if cls == 4:
            class_name += " (Cable)"
        elif cls == 5:
            class_name += " (Connector)"
        
        class_labels.append(f"{class_name}: {pixel_count} pixels ({percentage:.2f}%)")
    
    # Create the figure
    plt.figure(figsize=(12, 8))
    
    # Original grayscale mask
    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap='gray')
    plt.title("Original Mask (Grayscale)")
    plt.axis('off')
    
    # Colored mask
    plt.subplot(1, 2, 2)
    plt.imshow(colored_mask)
    plt.title("Colored Mask")
    plt.axis('off')
    
    # Add class information
    plt.figtext(0.5, 0.01, "\n".join(class_labels), ha="center", fontsize=9,
                bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    # Add filename as overall title
    plt.suptitle(mask_basename)
    
    # Save the visualization
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for the class labels
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    return True

def main():
    """Main function to create visualizations for all masks"""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of mask files
    mask_pattern = os.path.join(args.mask_dir, args.file_pattern)
    mask_files = list(Path(args.mask_dir).glob(mask_pattern))
    
    print(f"Found {len(mask_files)} mask files matching the pattern.")
    
    # Process each mask
    successful = 0
    for mask_file in tqdm(mask_files, desc="Creating visualizations"):
        if visualize_mask(str(mask_file), args.output_dir, args.class_id):
            successful += 1
    
    print(f"Successfully created {successful} visualizations in {args.output_dir}")

if __name__ == "__main__":
    main()