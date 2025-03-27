import os
import cv2
import numpy as np
import random
import shutil
import argparse
from tqdm import tqdm
import albumentations as A

# How to use in cmd:
# python bolt.py --component bolt1 --target 650
# Default component name is 'bolt' and default count is 600 if unspecified

# Parse command line arguments
parser = argparse.ArgumentParser(description='Data augmentation for bolt components')
parser.add_argument('--component', type=str, default='bolt', 
                    help='Component name (e.g., bolt1, bolt2, bolt3)')
parser.add_argument('--target', type=int, default=600,
                    help='Target number of images (default: 600)')
args = parser.parse_args()

# Component name and target count from command line
COMPONENT_NAME = args.component
TARGET_COUNT = args.target

print(f"Running augmentation for component: {COMPONENT_NAME}")
print(f"Target count: {TARGET_COUNT} images")

# Directory settings
IMAGE_DIR = 'images'
MASK_DIR = 'segmented_images'
OUTPUT_IMAGE_DIR = 'augmented_images'  # Output directory for augmented images
OUTPUT_MASK_DIR = 'augmented_masks'    # Output directory for augmented masks

# Create output directories if they don't exist
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

# Minimum ratio of component pixels in augmented mask compared to original
MIN_COMPONENT_RATIO = 0.5

# Tracking dictionary to avoid duplicate transformations on the same image
augmentation_tracker = {}

def get_bolt_images():
    """Get all image paths that have corresponding bolt masks."""
    bolt_images = []
    for mask_file in os.listdir(MASK_DIR):
        if mask_file.endswith('_semantic_mask.png'):
            # Read the mask to check if it contains a bolt
            mask_path = os.path.join(MASK_DIR, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is not None and np.max(mask) > 0:  # Check if mask contains foreground
                # Get corresponding image file
                base_name = mask_file.replace('_semantic_mask.png', '')
                image_file = f"{base_name}.jpg"
                image_path = os.path.join(IMAGE_DIR, image_file)
                
                if os.path.exists(image_path):
                    bolt_images.append((image_path, mask_path, base_name))
    
    return bolt_images

def count_current_images():
    """Count the current number of bolt images."""
    return len(get_bolt_images())

def create_augmentation_types():
    """Create different augmentation pipelines suitable for semantic segmentation."""
    aug_types = {
        'rot': A.Compose([
            A.RandomRotate90(p=1.0),
        ]),
        'hflip': A.Compose([
            A.HorizontalFlip(p=1.0),
        ]),
        'vflip': A.Compose([
            A.VerticalFlip(p=1.0),
        ]),
        'bright': A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=1.0),
        ]),
        'blur': A.Compose([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ]),
        'noise': A.Compose([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        ]),
        'shift': A.Compose([
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0, rotate_limit=0, p=1.0),
        ]),
        'scale': A.Compose([
            A.ShiftScaleRotate(shift_limit=0, scale_limit=0.2, rotate_limit=0, p=1.0),
        ]),
        'rotate': A.Compose([
            A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=45, p=1.0),
        ]),
        'combo1': A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.8),
        ]),
        'combo2': A.Compose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=0, p=0.5),
        ]),
    }
    return aug_types

def validate_augmentation(original_mask, augmented_mask):
    """
    Validate that the augmented mask contains sufficient bolt pixels.
    Returns True if valid, False otherwise.
    """
    original_count = np.sum(original_mask > 0)
    augmented_count = np.sum(augmented_mask > 0)
    
    # If original has no bolt pixels, skip validation
    if original_count == 0:
        return False
    
    # Check if augmented mask has at least MIN_COMPONENT_RATIO of the original bolt pixels
    ratio = augmented_count / original_count
    return ratio >= MIN_COMPONENT_RATIO

def copy_original_files(bolt_images):
    """Copy original images and masks to the output directories."""
    print("Copying original files to output directories...")
    for image_path, mask_path, base_name in tqdm(bolt_images):
        # Copy image
        dest_image_path = os.path.join(OUTPUT_IMAGE_DIR, os.path.basename(image_path))
        shutil.copy2(image_path, dest_image_path)
        
        # Copy mask
        dest_mask_path = os.path.join(OUTPUT_MASK_DIR, os.path.basename(mask_path))
        shutil.copy2(mask_path, dest_mask_path)

def augment_images():
    """Main function to augment bolt images."""
    # Get all bolt images
    bolt_images = get_bolt_images()
    current_count = len(bolt_images)
    
    if not bolt_images:
        print("No bolt images found.")
        return
    
    # Calculate how many augmentations are needed
    num_augmentations_needed = TARGET_COUNT - current_count
    
    if num_augmentations_needed <= 0:
        print(f"Already have {current_count} images, which meets or exceeds the target of {TARGET_COUNT}.")
        # Still copy the original files to the output directories
        copy_original_files(bolt_images)
        return
    
    print(f"Found {current_count} bolt images. Need to generate {num_augmentations_needed} augmentations.")
    
    # Copy original files to output directories
    copy_original_files(bolt_images)
    
    # Create augmentation types
    aug_types = create_augmentation_types()
    aug_type_keys = list(aug_types.keys())
    
    # Counter for successful augmentations
    successful_augmentations = 0
    
    # Augmentation count per image
    augmentation_count = {}
    
    # Main augmentation loop
    with tqdm(total=num_augmentations_needed) as pbar:
        while successful_augmentations < num_augmentations_needed:
            # Select a random image to augment
            image_path, mask_path, base_name = random.choice(bolt_images)
            
            # Initialize augmentation count for this image if not already done
            if base_name not in augmentation_count:
                augmentation_count[base_name] = 0
            
            # Select a random augmentation type not yet applied to this image
            if base_name not in augmentation_tracker:
                augmentation_tracker[base_name] = set()
            
            available_aug_types = [aug for aug in aug_type_keys 
                                   if aug not in augmentation_tracker[base_name]]
            
            # If all augmentation types have been applied to this image, continue to next image
            if not available_aug_types:
                continue
            
            aug_type = random.choice(available_aug_types)
            augmentation = aug_types[aug_type]
            
            # Read image and mask
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None or mask is None:
                print(f"Error reading {image_path} or {mask_path}")
                continue
            
            # Apply augmentation
            augmented = augmentation(image=image, mask=mask)
            augmented_image = augmented['image']
            augmented_mask = augmented['mask']
            
            # Validate augmentation
            if not validate_augmentation(mask, augmented_mask):
                continue
            
            # Update augmentation tracker
            augmentation_tracker[base_name].add(aug_type)
            augmentation_count[base_name] += 1
            
            # Save augmented image and mask
            aug_img_file = f"{base_name}aug{aug_type}{augmentation_count[base_name]}.jpg"
            aug_mask_file = f"{base_name}aug{aug_type}{augmentation_count[base_name]}_semantic_mask.png"
            
            cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR, aug_img_file), augmented_image)
            cv2.imwrite(os.path.join(OUTPUT_MASK_DIR, aug_mask_file), augmented_mask)
            
            successful_augmentations += 1
            pbar.update(1)
            
            # Break if we've reached the target
            if successful_augmentations >= num_augmentations_needed:
                break
    
    print(f"Successfully generated {successful_augmentations} augmented bolt images.")
    print(f"Total bolt images now: {current_count + successful_augmentations}")
    print(f"Files are saved in '{OUTPUT_IMAGE_DIR}' and '{OUTPUT_MASK_DIR}' directories.")

def check_image_mask_correspondence():
    """Check that all images have corresponding masks and vice versa."""
    print("Checking image-mask correspondence in output directories...")
    
    # Get all image files
    image_files = [f.replace('.jpg', '') for f in os.listdir(OUTPUT_IMAGE_DIR) 
                   if f.endswith('.jpg')]
    
    # Get all mask files
    mask_files = [f.replace('_semantic_mask.png', '') for f in os.listdir(OUTPUT_MASK_DIR) 
                  if f.endswith('_semantic_mask.png')]
    
    # Check if all images have masks
    images_without_masks = [img for img in image_files if img not in mask_files]
    
    # Check if all masks have images
    masks_without_images = [mask for mask in mask_files if mask not in image_files]
    
    if images_without_masks:
        print(f"Warning: {len(images_without_masks)} images don't have corresponding masks.")
        print(f"Example: {images_without_masks[:3]}")
    
    if masks_without_images:
        print(f"Warning: {len(masks_without_images)} masks don't have corresponding images.")
        print(f"Example: {masks_without_images[:3]}")
        
    if not images_without_masks and not masks_without_images:
        print("All images have corresponding masks and vice versa. Perfect match!")
    
    print(f"Total images in output directory: {len(image_files)}")
    print(f"Total masks in output directory: {len(mask_files)}")

if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"AUGMENTATION PROCESS FOR {COMPONENT_NAME.upper()}")
    print(f"{'='*50}\n")
    augment_images()
    check_image_mask_correspondence()
    print(f"\nAugmentation complete for {COMPONENT_NAME}!")
    print(f"Target count: {TARGET_COUNT}")
    print(f"Results saved in '{OUTPUT_IMAGE_DIR}' and '{OUTPUT_MASK_DIR}'")