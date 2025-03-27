import os
import cv2
import numpy as np
import random
import shutil
from tqdm import tqdm
from pathlib import Path

# Constants
PLASTICCOVER_CLASS = 7  # Class ID for plasticcover
TARGET_COUNT = 1800  # Target count to match busbar class
VALIDATION_THRESHOLD = 0.5  # Minimum ratio of component pixels that must be preserved

# Create necessary directories
def create_directories():
    os.makedirs('augmented_images', exist_ok=True)
    os.makedirs('augmented_masks', exist_ok=True)

# Get all image paths containing the plasticcover component
def get_plasticcover_images():
    image_paths = []
    mask_paths = []
    
    for mask_file in os.listdir('segmented_images'):
        if not mask_file.endswith('_semantic_mask.png'):
            continue
            
        mask_path = os.path.join('segmented_images', mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Check if mask contains plasticcover class
        if np.any(mask == PLASTICCOVER_CLASS):
            base_name = mask_file.replace('_semantic_mask.png', '')
            image_file = f"{base_name}.jpg"
            image_path = os.path.join('images', image_file)
            
            if os.path.exists(image_path):
                image_paths.append(image_path)
                mask_paths.append(mask_path)
    
    return list(zip(image_paths, mask_paths))

# Augmentation functions
def rotate(image, mask, angle):
    """Rotate image and mask by a specified angle."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply same transformation to both image and mask
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    rotated_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    
    return rotated_image, rotated_mask

def flip(image, mask, direction):
    """Flip image and mask horizontally or vertically."""
    if direction == 'horizontal':
        flipped_image = cv2.flip(image, 1)
        flipped_mask = cv2.flip(mask, 1)
    else:  # vertical
        flipped_image = cv2.flip(image, 0)
        flipped_mask = cv2.flip(mask, 0)
    
    return flipped_image, flipped_mask

def adjust_brightness_contrast(image, mask, alpha, beta):
    """Adjust brightness and contrast of the image."""
    # Only adjust the image, not the mask
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image, mask.copy()

def add_noise(image, mask, noise_level):
    """Add Gaussian noise to the image."""
    # Generate Gaussian noise
    h, w, c = image.shape
    noise = np.random.normal(0, noise_level, (h, w, c)).astype(np.uint8)
    
    # Add noise to the image, not to the mask
    noisy_image = cv2.add(image, noise)
    return noisy_image, mask.copy()

def perspective_transform(image, mask, strength):
    """Apply perspective transform to simulate different viewing angles."""
    h, w = image.shape[:2]
    
    # Define source points
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    
    # Define destination points with random offsets for perspective change
    offset_range = int(w * strength)
    tl = [random.randint(0, offset_range), random.randint(0, offset_range)]
    tr = [w - random.randint(0, offset_range), random.randint(0, offset_range)]
    br = [w - random.randint(0, offset_range), h - random.randint(0, offset_range)]
    bl = [random.randint(0, offset_range), h - random.randint(0, offset_range)]
    
    dst_pts = np.float32([tl, tr, br, bl])
    
    # Get perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Apply transformation to both image and mask
    warped_image = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    warped_mask = cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    
    return warped_image, warped_mask

def shear(image, mask, shear_factor_x, shear_factor_y):
    """Apply shear transformation."""
    h, w = image.shape[:2]
    
    # Define shear matrix
    M = np.float32([
        [1, shear_factor_x, 0],
        [shear_factor_y, 1, 0]
    ])
    
    # Apply transformation to both image and mask
    sheared_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    sheared_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    
    return sheared_image, sheared_mask

def crop_and_resize(image, mask, crop_percentage):
    """Crop a portion of the image and resize it back to original dimensions."""
    h, w = image.shape[:2]
    
    # Calculate crop dimensions
    crop_h = int(h * (1 - crop_percentage))
    crop_w = int(w * (1 - crop_percentage))
    
    # Calculate random starting points for crop
    start_h = random.randint(0, h - crop_h)
    start_w = random.randint(0, w - crop_w)
    
    # Crop both image and mask
    cropped_image = image[start_h:start_h + crop_h, start_w:start_w + crop_w]
    cropped_mask = mask[start_h:start_h + crop_h, start_w:start_w + crop_w]
    
    # Resize back to original dimensions
    resized_image = cv2.resize(cropped_image, (w, h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(cropped_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    return resized_image, resized_mask

def elastic_transform(image, mask, alpha, sigma, random_state=None):
    """Apply elastic transformation to simulate deformation."""
    if random_state is None:
        random_state = np.random.RandomState(None)
        
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma) * alpha
    
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    
    distorted_image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    distorted_mask = map_coordinates(mask, indices, order=0, mode='reflect').reshape(mask.shape)
    
    return distorted_image, distorted_mask

# Since elastic_transform requires scipy, let's use our other transformations
def combined_transform(image, mask):
    """Apply a combination of transformations and return parameter details."""
    # Dictionary of transformation abbreviations
    abbr = {
        'rotate': 'r',
        'perspective': 'p',
        'brightness': 'b',
        'fliph': 'fh',
        'flipv': 'fv',
        'noise': 'n',
        'shear': 'sh',
        'crop': 'c'
    }
    
    # Parameters for tracking what was applied
    applied_transforms = {}
    
    # Apply rotation
    angle = round(random.uniform(-30, 30))
    image, mask = rotate(image, mask, angle)
    applied_transforms['rotate'] = angle
    
    # Apply random perspective transform
    persp_strength = round(random.uniform(5, 15))
    persp_strength_decimal = persp_strength / 100  # Convert to decimal for actual transform
    image, mask = perspective_transform(image, mask, persp_strength_decimal)
    applied_transforms['perspective'] = persp_strength
    
    # Randomly adjust brightness and contrast
    alpha = round(random.uniform(0.8, 1.2), 1)
    beta = random.randint(-30, 30)
    image, mask = adjust_brightness_contrast(image, mask, alpha, beta)
    applied_transforms['brightness'] = (alpha, beta)
    
    # Create parameter string dynamically based on applied transformations
    param_parts = []
    for transform_type, value in applied_transforms.items():
        if transform_type == 'brightness':
            alpha, beta = value
            param_parts.append(f"{abbr[transform_type]}{alpha}-{beta}")
        else:
            param_parts.append(f"{abbr[transform_type]}{value}")
    
    param_str = "_".join(param_parts)
    
    return image, mask, param_str

# Validate that augmented mask still contains enough of the component
def validate_augmentation(original_mask, augmented_mask, class_id):
    """Check if augmented mask retains enough pixels of the class."""
    original_pixel_count = np.sum(original_mask == class_id)
    augmented_pixel_count = np.sum(augmented_mask == class_id)
    
    if original_pixel_count == 0:
        return False
    
    # Calculate ratio of component pixels preserved
    ratio = augmented_pixel_count / original_pixel_count
    return ratio >= VALIDATION_THRESHOLD

# Main augmentation pipeline
def augment_dataset():
    create_directories()
    
    print("Finding images with plasticcover component...")
    plasticcover_data = get_plasticcover_images()
    print(f"Found {len(plasticcover_data)} images containing plasticcover component.")
    
    # Calculate how many augmentations needed per image
    original_count = len(plasticcover_data)
    augmentations_needed = TARGET_COUNT - original_count
    augmentations_per_image = max(1, augmentations_needed // original_count + 1)
    
    print(f"Need to generate {augmentations_needed} augmented images.")
    print(f"Will create approximately {augmentations_per_image} augmentations per original image.")
    
    # Copy original images and masks to augmented folders
    for img_path, mask_path in plasticcover_data:
        img_filename = os.path.basename(img_path)
        mask_filename = os.path.basename(mask_path)
        
        shutil.copy(img_path, os.path.join('augmented_images', img_filename))
        shutil.copy(mask_path, os.path.join('augmented_masks', mask_filename))
    
        # Define augmentation types - we'll only use the type names now
    # The actual parameter values will be determined at runtime
    augmentation_types = [
        ('rotate', None),
        ('fliph', None),
        ('flipv', None),
        ('brightness', None),
        ('noise', None),
        ('perspective', None), 
        ('shear', None),
        ('crop', None),
        ('mix', None)  # Changed from 'combined' to 'mix'
    ]
    
    # Track applied augmentations to avoid duplicates
    applied_augmentations = {}
    augmentation_count = 0
    
    print("Generating augmented images...")
    pbar = tqdm(total=augmentations_needed)
    
    while augmentation_count < augmentations_needed:
        # Select a random image
        img_path, mask_path = random.choice(plasticcover_data)
        base_name = Path(img_path).stem
        
        # Load image and mask
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Select a random augmentation type that hasn't been applied to this image
        if base_name not in applied_augmentations:
            applied_augmentations[base_name] = set()
        
        # Apply augmentation with specific parameters
        aug_type = None
        param_value = None
        
        # Try to find an augmentation that hasn't been applied with these specific parameters
        max_attempts = 10
        for _ in range(max_attempts):
            # Select random augmentation type
            aug_type, _ = random.choice(augmentation_types)
            
            # Generate specific parameter values based on augmentation type
            if aug_type == 'rotate':
                param_value = round(random.uniform(-45, 45))
            elif aug_type == 'fliph':
                param_value = 'h'
            elif aug_type == 'flipv':
                param_value = 'v'
            elif aug_type == 'brightness':
                alpha = round(random.uniform(0.7, 1.3), 1)
                beta = random.randint(-40, 40)
                param_value = f"{alpha}_{beta}"
            elif aug_type == 'noise':
                param_value = round(random.uniform(5, 20))
            elif aug_type == 'perspective':
                param_value = round(random.uniform(5, 15))
            elif aug_type == 'shear':
                x = round(random.uniform(-0.2, 0.2), 1)
                y = round(random.uniform(-0.2, 0.2), 1)
                param_value = f"{x}_{y}"
            elif aug_type == 'crop':
                param_value = round(random.uniform(10, 30))
            elif aug_type == 'mix':
                # For mix transforms, the specific parameters will be determined 
                # when the transformation is actually applied
                param_value = None  # placeholder, will be replaced with actual values
                
            # Create a unique identifier for this augmentation+parameter combination
            aug_identifier = f"{aug_type}_{param_value}"
            
            # Check if this specific augmentation hasn't been applied to this image
            if aug_identifier not in applied_augmentations[base_name]:
                applied_augmentations[base_name].add(aug_identifier)
                break
        else:
            # If we've tried max_attempts times and couldn't find a new augmentation, skip this image
            continue
        
        # Apply the selected augmentation with specific parameters
        if aug_type == 'rotate':
            augmented_image, augmented_mask = rotate(image, mask, param_value)
        elif aug_type == 'fliph':
            augmented_image, augmented_mask = flip(image, mask, 'horizontal')
        elif aug_type == 'flipv':
            augmented_image, augmented_mask = flip(image, mask, 'vertical')
        elif aug_type == 'brightness':
            alpha, beta = map(float, param_value.split('_'))
            augmented_image, augmented_mask = adjust_brightness_contrast(image, mask, alpha, int(beta))
        elif aug_type == 'noise':
            augmented_image, augmented_mask = add_noise(image, mask, param_value)
        elif aug_type == 'perspective':
            strength = param_value / 100  # Convert 5-15 to 0.05-0.15
            augmented_image, augmented_mask = perspective_transform(image, mask, strength)
        elif aug_type == 'shear':
            x, y = map(float, param_value.split('_'))
            augmented_image, augmented_mask = shear(image, mask, x, y)
        elif aug_type == 'crop':
            crop_percent = param_value / 100  # Convert 10-30 to 0.1-0.3
            augmented_image, augmented_mask = crop_and_resize(image, mask, crop_percent)
        elif aug_type == 'mix':
            augmented_image, augmented_mask, param_value = combined_transform(image, mask)
        
        # Validate augmentation
        if validate_augmentation(mask, augmented_mask, PLASTICCOVER_CLASS):
            # Create descriptive filenames with parameter values
            aug_img_file = f"{base_name}_{aug_type}_{param_value}.jpg"
            aug_mask_file = f"{base_name}_{aug_type}_{param_value}_semantic_mask.png"
            
            # Save augmented image and mask
            cv2.imwrite(os.path.join('augmented_images', aug_img_file), augmented_image)
            cv2.imwrite(os.path.join('augmented_masks', aug_mask_file), augmented_mask)
            
            augmentation_count += 1
            pbar.update(1)
            
            # Check if we've reached the target
            if augmentation_count >= augmentations_needed:
                break
    
    pbar.close()
    
    # Count final number of images
    final_image_count = len(os.listdir('augmented_images'))
    print(f"Augmentation complete. Generated {augmentation_count} new images.")
    print(f"Total dataset size: {final_image_count} images (original + augmented)")

if __name__ == "__main__":
    # Import scipy functions for elastic transform if needed
    try:
        from scipy.ndimage import gaussian_filter, map_coordinates
        # Add elastic transform to augmentation types in the augment_dataset function
    except ImportError:
        print("SciPy not found. Elastic transformation will not be available.")
    
    augment_dataset()