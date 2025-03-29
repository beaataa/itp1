#!/usr/bin/env python3
"""
Multi-Class Mask Alignment Debugger

This script creates visualizations of images with their corresponding masks 
to help debug alignment issues. It generates visualizations for each 
image-mask pair showing the original image, the mask, and overlays with all classes colored.

Usage:
  python debug.py --image-dir <image_directory> --mask-dir <mask_directory>
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from pathlib import Path

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Debug mask alignment issues for multiple classes')
    parser.add_argument('--image-dir', type=str, default='images', help='Directory containing the original images')
    parser.add_argument('--mask-dir', type=str, default='segmented_images', help='Directory containing the semantic masks')
    parser.add_argument('--output-dir', type=str, default='alignment_debug', help='Directory to save debug visualizations')
    parser.add_argument('--alpha', type=float, default=0.3, help='Transparency of the overlay (0.0 to 1.0)')
    parser.add_argument('--class-list', type=str, default='', help='Comma-separated list of class IDs to focus on (empty = all classes)')
    return parser.parse_args()

def create_color_map():
    """Create a color map for visualizing different semantic classes"""
    # Define a color map for different classes (BGR format for OpenCV)
    colors = {
        0: [0, 0, 0],      # Background: black
        1: [255, 0, 0],    # Class 1: blue
        2: [0, 255, 0],    # Class 2: green
        3: [0, 0, 255],    # Class 3: red
        4: [255, 0, 255],  # Class 4 (Cable): magenta
        5: [255, 255, 0],  # Class 5 (Connector): cyan
        6: [0, 255, 255],  # Class 6: yellow
        7: [128, 128, 255], # Class 7: light red
        8: [128, 255, 128], # Class 8: light green
        9: [255, 128, 128], # Class 9: light blue
        10: [192, 0, 192],  # Class 10: dark magenta
    }
    return colors

def find_matching_mask(image_file, mask_dir):
    """Find the corresponding mask file for an image file"""
    image_name = Path(image_file).stem
    mask_file = os.path.join(mask_dir, f"{image_name}_semantic_mask.png")
    
    if os.path.exists(mask_file):
        return mask_file
    
    # Try alternative naming patterns if standard one doesn't exist
    for mask_file in os.listdir(mask_dir):
        if image_name in mask_file and mask_file.endswith('_semantic_mask.png'):
            return os.path.join(mask_dir, mask_file)
    
    return None

def create_debug_visualization(image_path, mask_path, output_path, alpha, focus_classes=None):
    """Create a debug visualization for an image-mask pair with multiple classes"""
    # Load image and mask
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return False
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Could not load mask {mask_path}")
        return False
    
    # Check if mask and image have the same dimensions
    if image.shape[:2] != mask.shape:
        print(f"Warning: Image and mask dimensions don't match for {image_path}")
        print(f"  Image shape: {image.shape}, Mask shape: {mask.shape}")
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Get color map for the classes
    color_map = create_color_map()
    
    # Convert image to RGB for visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Find unique classes in the mask
    unique_classes = np.unique(mask)
    
    # If focus_classes is specified, filter the unique classes
    if focus_classes is not None and len(focus_classes) > 0:
        unique_classes = np.array([c for c in unique_classes if c in focus_classes])
    
    # Create overlay mask for all classes
    overlay_mask = np.zeros_like(image)
    for class_id in unique_classes:
        if class_id == 0:  # Skip background class
            continue
        # Apply color for this class
        color_bgr = color_map.get(class_id, [0, 0, 0])
        overlay_mask[mask == class_id] = color_bgr
    
    # Create overlay
    overlay = cv2.addWeighted(image, 1.0, overlay_mask, alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    # Create colored mask for visualization
    colored_mask = np.zeros_like(image)
    for class_id in unique_classes:
        if class_id == 0:  # Skip background or make it black
            continue
        color_bgr = color_map.get(class_id, [0, 0, 0])
        colored_mask[mask == class_id] = color_bgr
    colored_mask_rgb = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB)
    
    # Calculate statistics for each class
    total_pixels = mask.size
    class_stats = []
    for class_id in unique_classes:
        class_pixels = np.sum(mask == class_id)
        percentage = (class_pixels / total_pixels) * 100
        
        class_name = f"Class {class_id}"
        if class_id == 4:
            class_name = "Class 4 (Cable)"
        elif class_id == 5:
            class_name = "Class 5 (Connector)"
            
        color_rgb = [c/255 for c in color_map.get(class_id, [0, 0, 0])[::-1]]  # Convert BGR to RGB and normalize
        
        class_stats.append({
            'id': class_id,
            'name': class_name,
            'pixels': class_pixels,
            'percentage': percentage,
            'color': color_rgb
        })
    
    # Sort classes by pixel count (descending)
    class_stats = sorted(class_stats, key=lambda x: x['pixels'], reverse=True)
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis('off')
    
    # Colored mask
    plt.subplot(1, 3, 2)
    plt.imshow(colored_mask_rgb)
    plt.title("Segmentation Mask")
    plt.axis('off')
    
    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(overlay_rgb)
    plt.title("Overlay")
    plt.axis('off')
    
    # Add filenames
    img_filename = os.path.basename(image_path)
    mask_filename = os.path.basename(mask_path)
    
    # Create legend for the classes
    legend_elements = []
    for stat in class_stats:
        if stat['id'] == 0:  # Skip background
            continue
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                         markerfacecolor=stat['color'], markersize=10, 
                                         label=f"{stat['name']}: {stat['pixels']} px ({stat['percentage']:.1f}%)"))
    
    # Add legend if there are classes to show
    if legend_elements:
        plt.figlegend(handles=legend_elements, loc='lower center', ncol=min(3, len(legend_elements)))
    
    plt.suptitle(f"Image: {img_filename} | Mask: {mask_filename}")
    
    # Save the visualization
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for the legend
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True

def main():
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse class list if provided
    focus_classes = None
    if args.class_list:
        focus_classes = [int(c.strip()) for c in args.class_list.split(',')]
        print(f"Focusing on classes: {focus_classes}")
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(list(Path(args.image_dir).glob(f'*{ext}')))
    
    # Filter out mask files that might be in the image directory
    image_files = [f for f in image_files if not f.name.endswith('_semantic_mask.png')]
    
    if not image_files:
        print(f"No image files found in {args.image_dir}")
        return
    
    print(f"Found {len(image_files)} images. Processing...")
    
    # Process each image
    success_count = 0
    for img_file in tqdm(image_files):
        # Find corresponding mask
        mask_file = find_matching_mask(str(img_file), args.mask_dir)
        
        if mask_file:
            # Create output filename
            output_filename = f"debug_{img_file.stem}.png"
            output_path = os.path.join(args.output_dir, output_filename)
            
            # Create visualization
            if create_debug_visualization(str(img_file), mask_file, output_path, 
                                        args.alpha, focus_classes):
                success_count += 1
        else:
            print(f"Warning: No matching mask found for {img_file}")
    
    print(f"Created {success_count} debug visualizations in {args.output_dir}")
    if focus_classes:
        print(f"Visualized classes: {focus_classes}")
    else:
        print("Visualized all classes found in the masks")

if __name__ == "__main__":
    main()