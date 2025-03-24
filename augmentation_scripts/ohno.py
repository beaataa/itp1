import os
import numpy as np
from PIL import Image
import argparse
from collections import defaultdict
import glob

def analyze_segmentation_masks(directory, extension="png"):
    """
    Analyzes all segmentation mask images in the given directory and reports unique labels.
    
    Args:
        directory (str): Path to the directory containing segmentation mask images
        extension (str): File extension of the segmentation mask images (default: png)
        
    Returns:
        dict: Dictionary with filenames as keys and lists of unique labels as values
        dict: Dictionary with label counts across all images
    """
    # Get all mask files with the specified extension
    mask_files = glob.glob(os.path.join(directory, f"*.{extension}"))
    
    if not mask_files:
        print(f"No {extension} files found in {directory}")
        return {}, {}
    
    # Dictionaries to store results
    file_labels = {}
    global_label_count = defaultdict(int)
    
    for mask_file in mask_files:
        try:
            # Open and convert the image to a numpy array
            mask = np.array(Image.open(mask_file))
            
            # Find unique labels in the mask
            unique_labels = np.unique(mask).tolist()
            
            # Store unique labels for this file
            file_labels[os.path.basename(mask_file)] = unique_labels
            
            # Update global counter
            for label in unique_labels:
                global_label_count[label] += 1
                
            print(f"Processed: {os.path.basename(mask_file)}, Labels: {unique_labels}")
            
        except Exception as e:
            print(f"Error processing {mask_file}: {e}")
    
    return file_labels, dict(global_label_count)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Analyze segmentation mask images and report unique labels")
    parser.add_argument("--dir", type=str, default=".", help="Directory containing mask images")
    parser.add_argument("--ext", type=str, default="png", help="File extension of mask images")
    parser.add_argument("--output", type=str, help="Optional output file for saving results")
    
    # Parse arguments
    args = parser.parse_args()
    
    print(f"Analyzing segmentation masks in {args.dir} with extension .{args.ext}")
    
    # Analyze masks
    file_labels, global_label_count = analyze_segmentation_masks(args.dir, args.ext)
    
    # Report summary
    print("\n--- Summary ---")
    print(f"Processed {len(file_labels)} mask files")
    
    if global_label_count:
        print("\nUnique labels found across all images:")
        for label, count in sorted(global_label_count.items()):
            print(f"  Label {label}: found in {count} images")
    else:
        print("\nNo labels found in any images")
    
    # Write results to file if requested
    if args.output:
        try:
            with open(args.output, 'w') as f:
                f.write("--- Per-file labels ---\n")
                for filename, labels in file_labels.items():
                    f.write(f"{filename}: {labels}\n")
                
                f.write("\n--- Global label counts ---\n")
                for label, count in sorted(global_label_count.items()):
                    f.write(f"Label {label}: found in {count} images\n")
                    
            print(f"\nResults written to {args.output}")
        except Exception as e:
            print(f"Error writing to output file: {e}")

if __name__ == "__main__":
    main()