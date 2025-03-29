import os
import cv2
import numpy as np
import random
import shutil
from tqdm import tqdm
from collections import defaultdict
import albumentations as A
import matplotlib.pyplot as plt

# Constants
CABLE_CLASS = 4  # The class ID for cable
TARGET_COUNT = 1800  # Target number of cable samples
PIXEL_THRESHOLD = 0.5  # Minimum ratio of cable pixels after augmentation
INPUT_IMG_DIR = "images"
INPUT_MASK_DIR = "segmented_images"
OUTPUT_IMG_DIR = "augmented_images"
OUTPUT_MASK_DIR = "augmented_masks"

def setup_output_dirs():
    """Create output directories if they don't exist"""
    for dir_path in [OUTPUT_IMG_DIR, OUTPUT_MASK_DIR]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        else:
            # Empty the directory if it already exists
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)

def get_cable_images():
    """Find all images containing cables and their corresponding masks"""
    cable_images = []
    print("Scanning for images with cable components...")
    
    # Check each mask file
    for mask_file in tqdm(os.listdir(INPUT_MASK_DIR)):
        if not mask_file.endswith("_semantic_mask.png"):
            continue
            
        mask_path = os.path.join(INPUT_MASK_DIR, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Check if mask contains cable class
        if CABLE_CLASS in np.unique(mask):
            # Get corresponding image file name
            base_name = mask_file.replace("_semantic_mask.png", "")
            img_file = f"{base_name}.jpg"
            img_path = os.path.join(INPUT_IMG_DIR, img_file)
            
            # Verify image exists
            if os.path.exists(img_path):
                cable_images.append((base_name, img_path, mask_path))
    
    print(f"Found {len(cable_images)} images with cable components")
    return cable_images

def count_component_pixels(mask):
    """Count the number of pixels belonging to the cable class"""
    return np.sum(mask == CABLE_CLASS)

def create_augmentation_pipeline(aug_type, severity=None):
    """Create an augmentation pipeline based on specified type and severity"""
    
    # Define augmentation ranges
    rotate_range = (-45, 45)  # Wider rotation range
    scale_range = (0.7, 1.3)  # More varied scaling
    brightness_range = (-0.3, 0.3)  # Allows both darkening and brightening
    contrast_range = (-0.3, 0.3)  # Allows both decreasing and increasing contrast
    hue_range = (-20, 20)  # HSV adjustments for cable color variation
    perspective_range = (0.05, 0.15)  # Perspective effect
    
    # Set specific severity if provided
    if severity == "mild":
        rotate_range = (-15, 15)
        scale_range = (0.9, 1.1)
        brightness_range = (-0.1, 0.1)
        contrast_range = (-0.1, 0.1)
        hue_range = (-10, 10)
        perspective_range = (0.02, 0.05)
    elif severity == "strong":
        rotate_range = (-60, 60)
        scale_range = (0.6, 1.4)
        brightness_range = (-0.4, 0.4)
        contrast_range = (-0.4, 0.4)
        hue_range = (-30, 30)
        perspective_range = (0.1, 0.2)
    
    # Select augmentation based on type
    if aug_type == "rotate":
        angle = random.uniform(*rotate_range)
        return A.Compose([
            A.Rotate(limit=(angle, angle), p=1.0)
        ]), f"{int(angle)}"
        
    elif aug_type == "flip":
        flip_type = random.choice(["h", "v"])
        if flip_type == "h":
            return A.Compose([A.HorizontalFlip(p=1.0)]), "fh"
        else:
            return A.Compose([A.VerticalFlip(p=1.0)]), "fv"
            
    elif aug_type == "scale":
        scale = random.uniform(*scale_range)
        return A.Compose([
            A.Affine(scale=scale, p=1.0)
        ]), f"{scale:.2f}"
        
    elif aug_type == "brightness":
        # Choose a specific brightness and contrast adjustment
        brightness = random.uniform(*brightness_range)
        contrast = random.uniform(*contrast_range)
        
        return A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=(brightness, brightness),
                contrast_limit=(contrast, contrast),
                p=1.0
            )
        ]), f"{contrast:.1f}_{int(brightness*100)}"
        
    elif aug_type == "perspective":
        scale = random.uniform(*perspective_range)
        return A.Compose([
            A.Perspective(scale=scale, p=1.0)
        ]), f"{int(scale*100)}"
    
    elif aug_type == "hsv":
        hue = random.uniform(*hue_range)
        saturation = random.uniform(-0.2, 0.2)
        value = random.uniform(-0.2, 0.2)
        
        return A.Compose([
            A.HueSaturationValue(
                hue_shift_limit=(hue, hue),
                sat_shift_limit=(saturation*100, saturation*100),
                val_shift_limit=(value*100, value*100),
                p=1.0
            )
        ]), f"h{int(hue)}_s{int(saturation*100)}_v{int(value*100)}"
        
    elif aug_type == "mix":
        # Mixed transformation with rotation + perspective + brightness
        angle = random.uniform(*rotate_range)
        persp_scale = random.uniform(*perspective_range)
        brightness = random.uniform(*brightness_range)
        contrast = random.uniform(*contrast_range)
        
        return A.Compose([
            A.Rotate(limit=(angle, angle), p=1.0),
            A.Perspective(scale=persp_scale, p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=(brightness, brightness),
                contrast_limit=(contrast, contrast),
                p=1.0
            )
        ]), f"mix_r{int(angle)}_p{int(persp_scale*100)}_b{contrast:.1f}_{int(brightness*100)}"
        
    elif aug_type == "cable_combo":
        # Special combo for cables - rotation + HSV shift
        angle = random.uniform(*rotate_range)
        hue = random.uniform(*hue_range)
        scale = random.uniform(*scale_range)
        
        return A.Compose([
            A.Rotate(limit=(angle, angle), p=1.0),
            A.HueSaturationValue(
                hue_shift_limit=(hue, hue),
                sat_shift_limit=(-15, 15),
                val_shift_limit=(-10, 10),
                p=1.0
            ),
            A.Affine(scale=scale, p=1.0)
        ]), f"cable_r{int(angle)}_h{int(hue)}_s{scale:.2f}"
        
    else:
        # Default to a slight rotation as fallback
        return A.Compose([A.Rotate(limit=10, p=1.0)]), "default"

def apply_augmentation(image_data, target_count):
    """Apply augmentations to reach target count"""
    # Unpack image data
    base_names = [x[0] for x in image_data]
    img_paths = [x[1] for x in image_data]
    mask_paths = [x[2] for x in image_data]
    
    # Track augmentations applied to each image
    applied_augs = defaultdict(set)
    
    # First, copy all original images to output directories
    print("Copying original images to output directories...")
    for base_name, img_path, mask_path in zip(base_names, img_paths, mask_paths):
        img_file = os.path.basename(img_path)
        mask_file = os.path.basename(mask_path)
        
        # Copy to output dirs
        shutil.copy(img_path, os.path.join(OUTPUT_IMG_DIR, img_file))
        shutil.copy(mask_path, os.path.join(OUTPUT_MASK_DIR, mask_file))
    
    # Track original cable pixel counts
    original_pixel_counts = {}
    
    # Calculate how many more images we need
    current_count = len(image_data)
    augmentations_needed = target_count - current_count
    
    if augmentations_needed <= 0:
        print("No augmentations needed. Dataset already has enough cable images.")
        return
    
    print(f"Generating {augmentations_needed} augmented images...")
    
    # Available augmentation types (updated for cables)
    aug_types = ["rotate", "flip", "scale", "brightness", "perspective", "hsv", "mix", "cable_combo"]
    
    # Weight more towards cable-specific augmentations
    aug_weights = [0.15, 0.1, 0.15, 0.1, 0.1, 0.1, 0.15, 0.15]
    
    # Create augmentations until we reach the target
    progress_bar = tqdm(total=augmentations_needed)
    augmented_count = 0
    
    # Try to distribute augmentations evenly across source images
    augs_per_image = max(1, augmentations_needed // len(image_data) + 1)
    
    while augmented_count < augmentations_needed:
        # Select a random image to augment
        idx = random.randint(0, len(image_data) - 1)
        base_name, img_path, mask_path = image_data[idx]
        
        # Skip if this image already has enough augmentations
        if len(applied_augs[base_name]) >= augs_per_image:
            continue
            
        # Load image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Store original cable pixel count if not already calculated
        if base_name not in original_pixel_counts:
            original_pixel_counts[base_name] = count_component_pixels(mask)
            
        original_count = original_pixel_counts[base_name]
        
        # If original image has no cable pixels, skip it
        if original_count == 0:
            continue
            
        # Select augmentation type, weighted for cable-specific augmentations
        aug_type = random.choices(aug_types, weights=aug_weights, k=1)[0]
        
        # Try to avoid repeating the same augmentation type on the same image
        attempts = 0
        while aug_type in applied_augs[base_name] and attempts < 5:
            aug_type = random.choices(aug_types, weights=aug_weights, k=1)[0]
            attempts += 1
            
        if aug_type in applied_augs[base_name]:
            continue  # Skip if we can't find a new augmentation type after attempts
        
        # Create augmentation with random parameters
        severity = random.choice([None, "mild", "strong"])
        aug_pipeline, param_str = create_augmentation_pipeline(aug_type, severity)
        
        # Apply augmentation
        augmented = aug_pipeline(image=image, mask=mask)
        aug_image = augmented['image']
        aug_mask = augmented['mask']
        
        # Convert back to BGR for saving
        aug_image = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
        
        # Check if the cable is still present with sufficient pixels
        aug_count = count_component_pixels(aug_mask)
        if aug_count < PIXEL_THRESHOLD * original_count:
            # Skip this augmentation as it reduced the cable too much
            continue
            
        # Create output filenames
        aug_img_file = f"{base_name}_{aug_type}_{param_str}.jpg"
        aug_mask_file = f"{base_name}_{aug_type}_{param_str}_semantic_mask.png"
        
        # Save augmented files
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, aug_img_file), aug_image)
        cv2.imwrite(os.path.join(OUTPUT_MASK_DIR, aug_mask_file), aug_mask)
        
        # Update tracking
        applied_augs[base_name].add(aug_type)
        augmented_count += 1
        progress_bar.update(1)
        
        # Break if we've reached our target
        if augmented_count >= augmentations_needed:
            break
            
    progress_bar.close()
    print(f"Generated {augmented_count} augmented images.")
    print(f"Total cable images now: {current_count + augmented_count}")
    
    # Generate augmentation statistics
    print("\nAugmentation Statistics:")
    aug_type_counts = {}
    for base_name, augs in applied_augs.items():
        for aug in augs:
            if aug not in aug_type_counts:
                aug_type_counts[aug] = 0
            aug_type_counts[aug] += 1
    
    for aug_type, count in sorted(aug_type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {aug_type}: {count}")

def visualize_samples(num_samples=5):
    """Visualize some sample augmentations"""
    all_images = os.listdir(OUTPUT_IMG_DIR)
    aug_images = [img for img in all_images if "_" in img and not img.endswith("_semantic_mask.png")]
    
    if not aug_images:
        print("No augmented images to visualize")
        return
        
    # Select random samples
    samples = random.sample(aug_images, min(num_samples, len(aug_images)))
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3*num_samples))
    
    for i, img_file in enumerate(samples):
        # Get corresponding mask
        base_parts = img_file.rsplit(".", 1)[0]
        mask_file = f"{base_parts}_semantic_mask.png"
        
        # Load image and mask
        img_path = os.path.join(OUTPUT_IMG_DIR, img_file)
        mask_path = os.path.join(OUTPUT_MASK_DIR, mask_file)
        
        if not os.path.exists(mask_path):
            print(f"Mask not found for {img_file}")
            continue
            
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Create visualization mask (highlight cable in blue)
        vis_mask = np.zeros_like(image)
        vis_mask[mask == CABLE_CLASS] = [0, 0, 255]  # Blue for cable
        
        # Overlay with transparency
        overlay = cv2.addWeighted(image, 0.7, vis_mask, 0.3, 0)
        
        # Plot
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"Augmented Image: {img_file}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title("Cable Overlay")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig("cable_augmentation_samples.png")
    print("Visualization saved to 'cable_augmentation_samples.png'")

def main():
    """Main function to run the augmentation pipeline"""
    print("Starting cable data augmentation...")
    
    # Setup output directories
    setup_output_dirs()
    
    # Get list of images with cables
    cable_images = get_cable_images()
    
    if not cable_images:
        print("No images with cable components found!")
        return
        
    # Apply augmentations to reach target count
    apply_augmentation(cable_images, TARGET_COUNT)
    
    # Visualize some samples
    visualize_samples()
    
    print("Augmentation complete!")

if __name__ == "__main__":
    main()