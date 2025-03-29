import os
import numpy as np
from PIL import Image
import glob

def remap_segmentation_masks(input_dir, output_dir, old_class=7, new_class=8):
    """
    Remaps segmentation masks by changing all pixels with value old_class to new_class.
    All images in the dataset contain only classes 0 (background) and old_class.
    
    Args:
        input_dir (str): Directory containing the original segmentation masks
        output_dir (str): Directory where remapped masks will be saved
        old_class (int): The class ID to be remapped (default: 7)
        new_class (int): The new class ID (default: 8)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all mask files (assuming they have common image extensions)
    mask_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
        mask_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    print(f"Found {len(mask_files)} mask files")
    
    for mask_file in mask_files:
        # Get the filename
        filename = os.path.basename(mask_file)
        
        # Load the mask
        mask = np.array(Image.open(mask_file))
        
        # Count original pixels of class 7
        original_count = np.sum(mask == old_class)
        
        # Create a new mask where old_class is replaced with new_class
        new_mask = np.where(mask == old_class, new_class, mask)
        
        # Verify the remapping
        remapped_count = np.sum(new_mask == new_class)
        
        if original_count == remapped_count:
            print(f"Successfully remapped {original_count} pixels in {filename}")
        else:
            print(f"Warning: Pixel count mismatch in {filename}!")
            print(f"Original class {old_class} pixels: {original_count}")
            print(f"Remapped to class {new_class}: {remapped_count}")
        
        # Save the new mask
        Image.fromarray(new_mask.astype(mask.dtype)).save(os.path.join(output_dir, filename))
    
    print("Remapping complete!")

# Example usage
if __name__ == "__main__":
    # Change these paths to match your dataset location
    input_directory = r"C:\Users\Ng Wei Herng\Downloads\SIT\Y2T2\ITP\semantic_aug\plasticcover\augmented_masks"
    output_directory = r"C:\Users\Ng Wei Herng\Downloads\SIT\Y2T2\ITP\semantic_aug\plasticcover\augmented_masks_remap"
    
    remap_segmentation_masks(input_directory, output_directory)