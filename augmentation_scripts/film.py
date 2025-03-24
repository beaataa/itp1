import os
import cv2
import numpy as np
import random
import shutil
from tqdm import tqdm
import albumentations as A
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Configuration
IMAGE_DIR = 'images'
MASK_DIR = 'segmented_images'
OUTPUT_IMAGE_DIR = 'augmented_images'
OUTPUT_MASK_DIR = 'augmented_masks'
PLASTICFILM_CLASS = 7
TARGET_COUNT = 700  # More reasonable target count from limited source images

# Create output directories if they don't exist
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

# Helper function to check if a mask contains the target class
def contains_class(mask, class_id):
    return np.any(mask == class_id)

# Helper function to extract base name from file
def get_base_name(filename):
    # Just get the name without the extension
    return os.path.splitext(filename)[0].replace('-combined-mask', '')

# Define augmentation strategies suitable for semantic segmentation
def get_augmentation_transforms():
    """Returns a dictionary of augmentation transforms suitable for semantic segmentation.
    Creates more diverse augmentations to handle the large number of variations needed.
    Each transform type has a unique "signature" to avoid exact duplication."""
    transforms = {
        'flip_h': A.Compose([
            A.HorizontalFlip(p=1.0),
        ]),
        
        'flip_v': A.Compose([
            A.VerticalFlip(p=1.0),
        ]),
        
        'rotate90': A.Compose([
            A.RandomRotate90(p=1.0),
        ]),
        
        'rotate_small': A.Compose([
            A.Rotate(limit=15, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0),
        ]),
        
        'rotate_medium': A.Compose([
            A.Rotate(limit=(20, 30), p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0),
        ]),
        
        'rotate_large': A.Compose([
            A.Rotate(limit=(30, 45), p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0),
        ]),
        
        'scale_up': A.Compose([
            A.RandomScale(scale_limit=(0.1, 0.2), p=1.0),
            A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32, 
                          border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        ]),
        
        'scale_down': A.Compose([
            A.RandomScale(scale_limit=(-0.2, -0.1), p=1.0),
            A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32, 
                          border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        ]),
        
        'shift_x': A.Compose([
            A.ShiftScaleRotate(shift_limit_x=0.15, shift_limit_y=0, scale_limit=0, rotate_limit=0, p=1.0, 
                               border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        ]),
        
        'shift_y': A.Compose([
            A.ShiftScaleRotate(shift_limit_x=0, shift_limit_y=0.15, scale_limit=0, rotate_limit=0, p=1.0, 
                               border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        ]),
        
        'shift_xy': A.Compose([
            A.ShiftScaleRotate(shift_limit=0.12, scale_limit=0, rotate_limit=0, p=1.0, 
                               border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        ]),
        
        'brightness_low': A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(0.05, 0.15), contrast_limit=0, p=1.0),
        ]),
        
        'brightness_high': A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(0.15, 0.25), contrast_limit=0, p=1.0),
        ]),
        
        'contrast_low': A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(0.05, 0.15), p=1.0),
        ]),
        
        'contrast_high': A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(0.15, 0.3), p=1.0),
        ]),
        
        'bright_contrast': A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
        ]),
        
        'blur_light': A.Compose([
            A.GaussianBlur(blur_limit=(3, 3), p=1.0),
        ]),
        
        'blur_medium': A.Compose([
            A.GaussianBlur(blur_limit=(5, 5), p=1.0),
        ]),
        
        'noise_light': A.Compose([
            A.GaussNoise(var_limit=(10, 30), p=1.0),
        ]),
        
        'noise_medium': A.Compose([
            A.GaussNoise(var_limit=(30, 50), p=1.0),
        ]),
        
        'cutout_small': A.Compose([
            A.CoarseDropout(max_holes=5, max_height=8, max_width=8, min_holes=3, 
                           min_height=4, min_width=4, fill_value=0, mask_fill_value=0, p=1.0)
        ]),
        
        'cutout_medium': A.Compose([
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=4, 
                           min_height=8, min_width=8, fill_value=0, mask_fill_value=0, p=1.0)
        ]),
        
        'combined1': A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
            A.RandomScale(scale_limit=0.15, p=0.5),
        ]),
        
        'combined2': A.Compose([
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=20, p=0.8, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ]),
        
        'combined3': A.Compose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.8,
                              border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
            A.RandomBrightnessContrast(p=0.5),
        ]),
        
        'combined4': A.Compose([
            A.RandomRotate90(p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.GaussNoise(var_limit=(5, 30), p=0.5),
        ]),
        
        'combined5': A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.7),
        ]),
        
        'combined6': A.Compose([
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=-0.1, rotate_limit=10, p=0.8,
                              border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
            A.GaussianBlur(blur_limit=3, p=0.5),
        ]),
    }
    return transforms


def main():
    print("="*80)
    print("SEMANTIC SEGMENTATION DATA AUGMENTATION")
    print("Target Class: Plasticfilm (Class ID 7)")
    print("="*80)
    
    # Get list of all image files
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
    print(f"Found {len(image_files)} images")
    
    # Identify images with plasticfilm class
    plasticfilm_images = []
    plasticfilm_only_images = []
    
    for img_file in tqdm(image_files, desc="Finding images with plasticfilm"):
        base_name = get_base_name(img_file)
        mask_file = f"{base_name}-combined-mask.png"
        mask_path = os.path.join(MASK_DIR, mask_file)
        
        if not os.path.exists(mask_path):
            continue
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if contains_class(mask, PLASTICFILM_CLASS):
            plasticfilm_images.append(img_file)
            
            # Check if the mask contains only plasticfilm and background
            unique_classes = np.unique(mask)
            if len(unique_classes) <= 2 and PLASTICFILM_CLASS in unique_classes:
                plasticfilm_only_images.append(img_file)
    
    print(f"Found {len(plasticfilm_images)} images containing plasticfilm")
    print(f"    Of which {len(plasticfilm_only_images)} only contain plasticfilm (single class)")

    # Calculate how many augmentations per image we need
    current_count = len(plasticfilm_images)
    if current_count == 0:
        print("No images with plasticfilm class found!")
        return
    
    # Calculate augmentations needed
    augmentations_needed = TARGET_COUNT - current_count
    if augmentations_needed <= 0:
        print(f"Current count ({current_count}) already exceeds target ({TARGET_COUNT}). No augmentation needed.")
        return
    
    augmentations_per_image = max(1, augmentations_needed // current_count)
    extra_augmentations = augmentations_needed % current_count
    
    print(f"Need to create {augmentations_needed} augmentations")
    print(f"Will create {augmentations_per_image} augmentations per image, plus {extra_augmentations} extra")
    
    # Get augmentation transforms
    transforms = get_augmentation_transforms()
    transform_types = list(transforms.keys())
    
    # Track which augmentations have been applied to each image
    image_augmentations = {img: [] for img in plasticfilm_images}
    
    # Copy original images to output directory first
    print("Copying original images to output directory...")
    for img_file in tqdm(plasticfilm_images):
        base_name = get_base_name(img_file)
        img_path = os.path.join(IMAGE_DIR, img_file)
        mask_path = os.path.join(MASK_DIR, f"{base_name}-combined-mask.png")
        
        # Copy original image and mask
        shutil.copy(img_path, os.path.join(OUTPUT_IMAGE_DIR, img_file))
        shutil.copy(mask_path, os.path.join(OUTPUT_MASK_DIR, f"{base_name}_semantic_mask.png"))
    
    # Now create augmentations
    augmentation_count = 0
    print("Creating augmentations...")
    
    # Calculate a more aggressive augmentation strategy for single-class images
    # since we need to create many variations from few source images
    single_class_per_image = max(5, (augmentations_needed // 2) // len(plasticfilm_only_images)) if plasticfilm_only_images else 0
    multi_class_per_image = max(1, (augmentations_needed // 2) // len([img for img in plasticfilm_images if img not in plasticfilm_only_images])) if len(plasticfilm_images) > len(plasticfilm_only_images) else 0
    
    print(f"Will create approximately {single_class_per_image} augmentations per single-class image")
    print(f"Will create approximately {multi_class_per_image} augmentations per multi-class image")
    
    # First apply augmentations to single-class images (to boost plasticfilm without affecting other components)
    for img_file in tqdm(plasticfilm_only_images, desc="Augmenting single-class images"):
        base_name = get_base_name(img_file)
        img_path = os.path.join(IMAGE_DIR, img_file)
        mask_path = os.path.join(MASK_DIR, f"{base_name}-combined-mask.png")
        
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue
        
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply augmentations more aggressively to single-class images
        for i in range(single_class_per_image):
            # Generate a unique transform signature for this specific augmentation
            if i % 4 == 0:  # Every 4th augmentation, create a truly random combination
                # Create a custom pipeline of 2-3 transforms for this specific image
                pipeline = []
                transform_categories = {
                    'flip': ['flip_h', 'flip_v'],
                    'rotate': ['rotate_small', 'rotate_medium', 'rotate_large', 'rotate90'],
                    'scale': ['scale_up', 'scale_down'],
                    'shift': ['shift_x', 'shift_y', 'shift_xy'],
                    'appearance': ['brightness_low', 'brightness_high', 'contrast_low', 
                                  'contrast_high', 'bright_contrast'],
                    'noise': ['blur_light', 'blur_medium', 'noise_light', 'noise_medium']
                }
                
                # Select 2-3 categories without replacement
                selected_categories = random.sample(list(transform_categories.keys()), k=random.randint(2, 3))
                
                # For each selected category, pick one transform
                selected_transforms = []
                for category in selected_categories:
                    selected_transforms.append(random.choice(transform_categories[category]))
                
                # Create a custom transform name
                aug_type = f"custom_{'_'.join(selected_categories)}_{augmentation_count}"
                
                # Create the custom transform pipeline
                custom_pipeline = []
                for transform_name in selected_transforms:
                    # Extract the underlying Albumentations transform from the existing composed transform
                    original_transform = transforms[transform_name]
                    # Get the internal transform (assuming single-transform compositions)
                    for transform in original_transform:
                        custom_pipeline.append(transform)
                
                # Create a new composed transform
                custom_transform = A.Compose(custom_pipeline)
                transform = custom_transform
            else:
                # Select a predefined transform with preference for combined transforms
                transform_types_weighted = transform_types + ['combined1', 'combined2', 'combined3', 'combined4', 'combined5', 'combined6']
                aug_type = random.choice(transform_types_weighted)
                transform = transforms[aug_type]
            
            # Apply the augmentation
            augmented = transform(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']
            
            # Verify the augmented mask still contains the target class
            if not contains_class(aug_mask, PLASTICFILM_CLASS):
                # If the transformation lost our target class, retry with a safer transform
                print(f"Warning: Augmentation {aug_type} lost target class. Trying safer transform...")
                safe_transform = transforms['flip_h']  # Use a very safe transform
                augmented = safe_transform(image=image, mask=mask)
                aug_image = augmented['image']
                aug_mask = augmented['mask']
                aug_type = 'safe_flip'
            
            # Save augmented image and mask
            aug_img_file = f"{base_name}_aug_{aug_type}_{augmentation_count}.jpg"
            aug_mask_file = f"{base_name}_aug_{aug_type}_{augmentation_count}_semantic_mask.png"
            
            cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR, aug_img_file), aug_image)
            cv2.imwrite(os.path.join(OUTPUT_MASK_DIR, aug_mask_file), aug_mask)
            
            # Track this augmentation
            image_augmentations[img_file].append(aug_type)
            augmentation_count += 1
            
            # Check if we've reached the target
            if augmentation_count >= augmentations_needed:
                break
                
        if augmentation_count >= augmentations_needed:
            break
    
    # Then apply augmentations to multi-class images if needed
    if augmentation_count < augmentations_needed:
        multi_class_images = [img for img in plasticfilm_images if img not in plasticfilm_only_images]
        for img_file in tqdm(multi_class_images, desc="Augmenting multi-class images"):
            base_name = get_base_name(img_file)
            img_path = os.path.join(IMAGE_DIR, img_file)
            mask_path = os.path.join(MASK_DIR, f"{base_name}-combined-mask.png")
            
            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                continue
            
            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Apply augmentations
            for i in range(multi_class_per_image):
                # Select a transform that hasn't been used for this image yet
                available_transforms = [t for t in transform_types if t not in image_augmentations[img_file]]
                
                # If all transforms have been used, allow reuse but try to minimize duplicates
                if not available_transforms:
                    transform_counts = {t: image_augmentations[img_file].count(t) for t in transform_types}
                    min_count = min(transform_counts.values())
                    available_transforms = [t for t, count in transform_counts.items() if count == min_count]
                
                aug_type = random.choice(available_transforms)
                transform = transforms[aug_type]
                
                # Apply the augmentation
                augmented = transform(image=image, mask=mask)
                aug_image = augmented['image']
                aug_mask = augmented['mask']
                
                # Verify the augmented mask still contains the target class
                if not contains_class(aug_mask, PLASTICFILM_CLASS):
                    # If the transformation lost our target class, retry with a safer transform
                    print(f"Warning: Augmentation {aug_type} lost target class. Trying safer transform...")
                    safe_transform = transforms['flip_h']  # Use a very safe transform
                    augmented = safe_transform(image=image, mask=mask)
                    aug_image = augmented['image']
                    aug_mask = augmented['mask']
                    aug_type = 'safe_flip'
                
                # Save augmented image and mask
                aug_img_file = f"{base_name}_aug_{aug_type}_{augmentation_count}.jpg"
                aug_mask_file = f"{base_name}_aug_{aug_type}_{augmentation_count}_semantic_mask.png"
                
                cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR, aug_img_file), aug_image)
                cv2.imwrite(os.path.join(OUTPUT_MASK_DIR, aug_mask_file), aug_mask)
                
                # Track this augmentation
                image_augmentations[img_file].append(aug_type)
                augmentation_count += 1
                
                # Check if we've reached the target
                if augmentation_count >= augmentations_needed:
                    break
            
            if augmentation_count >= augmentations_needed:
                break
    
    # Add extra augmentations if needed, prioritizing single-class images 
    if augmentation_count < augmentations_needed:
        # Prioritize single-class images to boost plasticfilm without affecting other components
        priority_images = plasticfilm_only_images
        if not priority_images:
            priority_images = plasticfilm_images
        
        extra_needed = augmentations_needed - augmentation_count
        print(f"Creating {extra_needed} additional augmentations...")
        
        for i in range(extra_needed):
            img_file = random.choice(priority_images)
            base_name = get_base_name(img_file)
            img_path = os.path.join(IMAGE_DIR, img_file)
            mask_path = os.path.join(MASK_DIR, f"{base_name}-combined-mask.png")
            
            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Use combined transforms for more diversity
            aug_type = random.choice(['combined1', 'combined2', 'combined3', 'combined4', 'combined5', 'combined6'])
            transform = transforms[aug_type]
            
            # Apply the augmentation
            augmented = transform(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']
            
            # Verify the augmented mask still contains the target class
            if not contains_class(aug_mask, PLASTICFILM_CLASS):
                # If the transformation lost our target class, retry with a safer transform
                print(f"Warning: Augmentation {aug_type} lost target class. Trying safer transform...")
                safe_transform = transforms['flip_h']  # Use a very safe transform
                augmented = safe_transform(image=image, mask=mask)
                aug_image = augmented['image']
                aug_mask = augmented['mask']
                aug_type = 'safe_flip'
            
            # Save augmented image and mask
            aug_img_file = f"{base_name}_aug_{aug_type}_{augmentation_count}.jpg"
            aug_mask_file = f"{base_name}_aug_{aug_type}_{augmentation_count}_semantic_mask.png"
            
            cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR, aug_img_file), aug_image)
            cv2.imwrite(os.path.join(OUTPUT_MASK_DIR, aug_mask_file), aug_mask)
            
            augmentation_count += 1
    
    # Final count
    final_count = current_count + augmentation_count
    print(f"Augmentation complete. Final count: {final_count}")
    print(f"Images saved to {OUTPUT_IMAGE_DIR}")
    print(f"Masks saved to {OUTPUT_MASK_DIR}")
    
    # Verify the augmented masks still contain the target class
    print("Verifying augmented masks contain the target class...")
    augmented_mask_files = [f for f in os.listdir(OUTPUT_MASK_DIR) if f.endswith('_semantic_mask.png')]
    
    valid_masks = 0
    for mask_file in tqdm(augmented_mask_files):
        mask_path = os.path.join(OUTPUT_MASK_DIR, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if contains_class(mask, PLASTICFILM_CLASS):
            valid_masks += 1
    
    print(f"Verification complete: {valid_masks}/{len(augmented_mask_files)} masks contain the target class")
    
    if valid_masks < len(augmented_mask_files):
        print("WARNING: Some augmented masks no longer contain the target class!")


if __name__ == "__main__":
    main()