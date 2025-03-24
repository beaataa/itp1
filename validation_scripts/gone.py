import os

def find_missing_segmented_images(images_folder, segmented_folder):
    """
    Find which images in the images_folder don't have corresponding segmented versions
    in the segmented_folder.
    
    Args:
        images_folder (str): Path to folder containing original images
        segmented_folder (str): Path to folder containing segmented images
    
    Returns:
        list: List of filenames (without extension) that don't have segmented versions
    """
    # Get list of all files in both directories
    image_files = os.listdir(images_folder)
    segmented_files = os.listdir(segmented_folder)
    
    # Extract basenames (without extension) from original images
    original_basenames = []
    for filename in image_files:
        # Split by the last dot to handle filenames with multiple dots
        basename = filename.rsplit('.', 1)[0]
        original_basenames.append(basename)
    
    # Extract basenames from segmented images and remove the '_semantic_mask' suffix
    segmented_basenames = []
    for filename in segmented_files:
        # Split by the last dot to handle filenames with multiple dots
        basename = filename.rsplit('.', 1)[0]
        # Remove '_semantic_mask' suffix
        if basename.endswith('_semantic_mask'):
            basename = basename[:-14]  # length of '_semantic_mask' is 14
            segmented_basenames.append(basename)
    
    # Find which original images don't have corresponding segmented versions
    missing_segmented = [basename for basename in original_basenames 
                         if basename not in segmented_basenames]
    
    return missing_segmented

if __name__ == "__main__":
    # Paths to your folders - update these to your actual paths
    images_folder = "images"
    segmented_folder = "segmented_images"
    
    missing_images = find_missing_segmented_images(images_folder, segmented_folder)
    
    # Print results
    print(f"Found {len(missing_images)} images without segmented versions:")
    for img in missing_images:
        print(f"  - {img}")
    
    # Optional: You can also save the results to a file
    with open("missing_segmented_images.txt", "w") as f:
        for img in missing_images:
            f.write(f"{img}\n")
    
    print(f"\nTotal original images: {len(os.listdir(images_folder))}")
    print(f"Total segmented images: {len(os.listdir(segmented_folder))}")
    print(f"Missing segmented images: {len(missing_images)}")