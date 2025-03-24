import os
import random
import cv2
import numpy as np
from tqdm import tqdm
import shutil
import albumentations as A

# Constants
CABLE_CLASS = 4  # Cable class in the semantic mask
TARGET_COUNT = 900  # Target number of augmented images to match busbar class
ORIGINAL_IMAGES_DIR = 'images'
MASKS_DIR = 'segmented_images'
OUTPUT_IMAGES_DIR = 'augmented_images/cable'
OUTPUT_MASKS_DIR = 'augmented_masks/cable'

# Create output directories if they don't exist
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASKS_DIR, exist_ok=True)

def get_cable_images():
    """Find all images that contain the cable component (class 4)"""
    cable_images = []
    
    for mask_file in os.listdir(MASKS_DIR):
        if not mask_file.endswith('_semantic_mask.png'):
            continue
        
        mask_path = os.path.join(MASKS_DIR, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Check if the mask contains the cable class
        if np.any(mask == CABLE_CLASS):
            # Get the corresponding image filename
            base_name = mask_file.replace('_semantic_mask.png', '')
            image_file = f"{base_name}.jpg"
            
            if os.path.exists(os.path.join(ORIGINAL_IMAGES_DIR, image_file)):
                cable_images.append((base_name, image_file, mask_file))
    
    return cable_images

def create_augmentations():
    """Define a set of augmentations suitable specifically for corrugated cable semantic segmentation"""
    augmentations = [
        # Horizontal and vertical flips - corrugated cables can appear in different orientations
        ('hflip', A.HorizontalFlip(p=1.0)),
        ('vflip', A.VerticalFlip(p=1.0)),
        
        # Rotations - corrugated cables are often curved or bent at various angles
        ('rot90', A.Rotate(limit=(90, 90), p=1.0)),
        ('rot180', A.Rotate(limit=(180, 180), p=1.0)),
        ('rot270', A.Rotate(limit=(270, 270), p=1.0)),
        
        # Moderate rotations - thick cables tend to curve rather than have sharp angle changes
        ('rot15', A.Rotate(limit=(-15, 15), p=1.0)),
        ('rot30', A.Rotate(limit=(-30, 30), p=1.0)),
        
        # Scale variations - corrugated cables appear at different distances and sizes
        ('scale_up', A.RandomScale(scale_limit=(0.1, 0.2), p=1.0)),
        ('scale_down', A.RandomScale(scale_limit=(-0.2, -0.1), p=1.0)),
        
        # Shifting/translation - cables can be positioned anywhere in panels
        ('shift', A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0, rotate_limit=0, p=1.0)),
        
        # Brightness and contrast for different lighting conditions - important for orange cables
        ('bright_up', A.RandomBrightnessContrast(brightness_limit=(0.1, 0.2), contrast_limit=0, p=1.0)),
        ('bright_down', A.RandomBrightnessContrast(brightness_limit=(-0.2, -0.1), contrast_limit=0, p=1.0)),
        ('contrast_up', A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(0.1, 0.2), p=1.0)),
        ('contrast_down', A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(-0.2, -0.1), p=1.0)),
        
        # Combined brightness and contrast adjustments
        ('bright_contrast', A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1.0)),
        
        # HSV shifts - to handle color variations in orange cables
        ('hue_shift', A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=1.0)),
        
        # Note: Shadow transformations removed as they may create unrealistic variations
        
        # Combinations specifically designed for corrugated cables
        ('cable_combo1', A.Compose([
            A.Rotate(limit=(-20, 20), p=1.0),
            A.RandomBrightnessContrast(brightness_limit=(0.05, 0.15), p=1.0)
        ])),
        ('cable_combo2', A.Compose([
            A.HorizontalFlip(p=1.0),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=1.0)
        ])),
        ('cable_combo3', A.Compose([
            A.Rotate(limit=(-15, 15), p=1.0),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=1.0)
        ])),
    ]
    
    return augmentations

def apply_augmentation(image, mask, transform):
    """Apply the given transformation to both image and mask"""
    transformed = transform(image=image, mask=mask)
    return transformed['image'], transformed['mask']

def main():
    # Get all images containing the cable component
    print("Finding images containing cable components...")
    cable_images = get_cable_images()
    original_count = len(cable_images)
    print(f"Found {original_count} images with cable components")
    
    if original_count == 0:
        print("No images with cable components found. Check your dataset and class label.")
        return
    
    # First, copy original images to output directories
    print("Copying original images to output directories...")
    for base_name, image_file, mask_file in tqdm(cable_images):
        img_path = os.path.join(ORIGINAL_IMAGES_DIR, image_file)
        mask_path = os.path.join(MASKS_DIR, mask_file)
        
        shutil.copy(img_path, os.path.join(OUTPUT_IMAGES_DIR, image_file))
        shutil.copy(mask_path, os.path.join(OUTPUT_MASKS_DIR, mask_file))
    
    # Calculate how many augmentations per image we need
    augmentations_needed = TARGET_COUNT - original_count
    if augmentations_needed <= 0:
        print(f"Already have {original_count} images, which meets or exceeds the target of {TARGET_COUNT}")
        return
    
    print(f"Need to generate {augmentations_needed} augmented images to reach target of {TARGET_COUNT}")
    
    # Create the augmentation transformations
    augmentation_list = create_augmentations()
    
    # Keep track of applied augmentations for each image
    applied_augmentations = {image_file: [] for _, image_file, _ in cable_images}
    
    # Generate augmentations
    total_augmented = 0
    pbar = tqdm(total=augmentations_needed, desc="Generating augmented images")
    
    while total_augmented < augmentations_needed:
        # Select a random image
        base_name, image_file, mask_file = random.choice(cable_images)
        
        # If we've used all augmentations for this image, skip it
        if len(applied_augmentations[image_file]) >= len(augmentation_list):
            continue
        
        # Select an augmentation not yet applied to this image
        available_augmentations = [aug for aug in augmentation_list if aug[0] not in applied_augmentations[image_file]]
        if not available_augmentations:
            continue
            
        aug_type, transform = random.choice(available_augmentations)
        
        # Add to applied augmentations for this image
        applied_augmentations[image_file].append(aug_type)
        
        # Read the image and mask
        image_path = os.path.join(ORIGINAL_IMAGES_DIR, image_file)
        mask_path = os.path.join(MASKS_DIR, mask_file)
        
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply the augmentation
        augmented_image, augmented_mask = apply_augmentation(image, mask, transform)
        
        # Verify the cable class is still present after augmentation
        if not np.any(augmented_mask == CABLE_CLASS):
            # If cable got cut off or removed by augmentation, skip this one
            applied_augmentations[image_file].remove(aug_type)
            continue
        
        # Save the augmented image and mask
        augmentation_count = total_augmented + 1  # Start count from 1
        aug_img_file = f"{base_name}_aug_{aug_type}_{augmentation_count}.jpg"
        aug_mask_file = f"{base_name}_aug_{aug_type}_{augmentation_count}_semantic_mask.png"
        
        cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, aug_img_file), augmented_image)
        cv2.imwrite(os.path.join(OUTPUT_MASKS_DIR, aug_mask_file), augmented_mask)
        
        total_augmented += 1
        pbar.update(1)
        
        if total_augmented >= augmentations_needed:
            break
    
    pbar.close()
    
    print(f"Augmentation complete! Generated {total_augmented} new images")
    print(f"Total dataset size for cable class: {original_count + total_augmented}")
    print(f"Augmented images saved to: {OUTPUT_IMAGES_DIR}")
    print(f"Augmented masks saved to: {OUTPUT_MASKS_DIR}")
    
    # Generate augmentation statistics
    print("\nAugmentation Statistics:")
    aug_type_counts = {}
    for img_augs in applied_augmentations.values():
        for aug in img_augs:
            if aug not in aug_type_counts:
                aug_type_counts[aug] = 0
            aug_type_counts[aug] += 1
    
    for aug_type, count in sorted(aug_type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {aug_type}: {count}")

if __name__ == "__main__":
    main()