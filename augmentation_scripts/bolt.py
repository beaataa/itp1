import os
import random
import numpy as np
import cv2
from tqdm import tqdm
import shutil
from pathlib import Path

def create_directories(base_dir, component_name):
    augmented_img_dir = os.path.join(base_dir, f'augmented_images/{component_name}')
    augmented_mask_dir = os.path.join(base_dir, f'augmented_masks/{component_name}')
    os.makedirs(augmented_img_dir, exist_ok=True)
    os.makedirs(augmented_mask_dir, exist_ok=True)
    return augmented_img_dir, augmented_mask_dir

def get_component_files(base_dir, component_name):
    img_dir = os.path.join(base_dir, 'images')
    mask_dir = os.path.join(base_dir, 'segmented_images')
    all_img_files = os.listdir(img_dir)
    all_mask_files = os.listdir(mask_dir)
    mask_to_img_mapping = {}
    for mask_file in all_mask_files:
        if not mask_file.endswith('.png'):
            continue
        base_name = mask_file.replace('_semantic_mask.png', '')
        img_file = f"{base_name}.jpg"
        if img_file in all_img_files:
            mask_to_img_mapping[mask_file] = img_file
    component_files = []
    for mask_file, img_file in mask_to_img_mapping.items():
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None and np.max(mask) > 0:
            component_files.append((img_file, mask_file))
    print(f"Found {len(component_files)} files for component '{component_name}'")
    return component_files

def apply_augmentations(img, mask, aug_types):
    augmented_pairs = []
    height, width = img.shape[:2]
    for aug_type in aug_types:
        aug_img = img.copy()
        aug_mask = mask.copy()
        if aug_type == 'horizontal_flip':
            aug_img = cv2.flip(aug_img, 1)
            aug_mask = cv2.flip(aug_mask, 1)
        elif aug_type == 'vertical_flip':
            aug_img = cv2.flip(aug_img, 0)
            aug_mask = cv2.flip(aug_mask, 0)
        elif aug_type == 'rotation_90':
            aug_img = cv2.rotate(aug_img, cv2.ROTATE_90_CLOCKWISE)
            aug_mask = cv2.rotate(aug_mask, cv2.ROTATE_90_CLOCKWISE)
        elif aug_type == 'rotation_180':
            aug_img = cv2.rotate(aug_img, cv2.ROTATE_180)
            aug_mask = cv2.rotate(aug_mask, cv2.ROTATE_180)
        elif aug_type == 'rotation_270':
            aug_img = cv2.rotate(aug_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            aug_mask = cv2.rotate(aug_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif aug_type == 'brightness':
            value = random.uniform(0.7, 1.3)
            hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v = cv2.convertScaleAbs(v, alpha=value)
            hsv = cv2.merge([h, s, v])
            aug_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        elif aug_type == 'contrast':
            value = random.uniform(0.7, 1.3)
            aug_img = cv2.convertScaleAbs(aug_img, alpha=value)
        elif aug_type == 'small_rotation':
            angle = random.uniform(-30, 30)
            M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
            aug_img = cv2.warpAffine(aug_img, M, (width, height), borderMode=cv2.BORDER_REFLECT)
            aug_mask = cv2.warpAffine(aug_mask, M, (width, height), borderMode=cv2.BORDER_REFLECT)
        elif aug_type == 'translation':
            tx = random.randint(-width//8, width//8)
            ty = random.randint(-height//8, height//8)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            aug_img = cv2.warpAffine(aug_img, M, (width, height), borderMode=cv2.BORDER_REFLECT)
            aug_mask = cv2.warpAffine(aug_mask, M, (width, height), borderMode=cv2.BORDER_REFLECT)
        elif aug_type == 'scale':
            scale = random.uniform(0.8, 1.2)
            M = cv2.getRotationMatrix2D((width/2, height/2), 0, scale)
            aug_img = cv2.warpAffine(aug_img, M, (width, height), borderMode=cv2.BORDER_REFLECT)
            aug_mask = cv2.warpAffine(aug_mask, M, (width, height), borderMode=cv2.BORDER_REFLECT)
        elif aug_type == 'shear':
            shear = random.uniform(-0.2, 0.2)
            M = np.float32([[1, shear, 0], [0, 1, 0]])
            aug_img = cv2.warpAffine(aug_img, M, (width, height), borderMode=cv2.BORDER_REFLECT)
            aug_mask = cv2.warpAffine(aug_mask, M, (width, height), borderMode=cv2.BORDER_REFLECT)
        _, aug_mask = cv2.threshold(aug_mask, 127, 255, cv2.THRESH_BINARY)
        augmented_pairs.append((aug_img, aug_mask, aug_type))
    return augmented_pairs

def safe_fallback_transform(img, mask, min_component_ratio, max_attempts=3):
    height, width = img.shape[:2]
    orig_count = np.count_nonzero(mask)
    for attempt in range(max_attempts):
        angle = random.uniform(-5, 5)
        # Skip near-zero angles to avoid pure duplicates.
        if abs(angle) < 1.0:
            continue
        safe_img = cv2.flip(img, 1)
        safe_mask = cv2.flip(mask, 1)
        M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        safe_img = cv2.warpAffine(safe_img, M, (width, height), borderMode=cv2.BORDER_REFLECT)
        safe_mask = cv2.warpAffine(safe_mask, M, (width, height), borderMode=cv2.BORDER_REFLECT)
        _, safe_mask = cv2.threshold(safe_mask, 127, 255, cv2.THRESH_BINARY)
        new_count = np.count_nonzero(safe_mask)
        if new_count >= min_component_ratio * orig_count:
            return safe_img, safe_mask, f"safe_flip_rot_{angle:.1f}"
    # If all attempts fail, return the last result.
    return safe_img, safe_mask, f"safe_flip_rot_{angle:.1f}"

def augment_component_data(base_dir, component_name, target_count, min_component_ratio=0.5):
    component_files = get_component_files(base_dir, component_name)
    original_count = len(component_files)
    num_to_generate = max(0, target_count - original_count)
    augmentations_per_image = num_to_generate // original_count + 1
    print(f"Original count: {original_count}, Target count: {target_count}")
    print(f"Need to generate {num_to_generate} augmented images")
    print(f"Will apply ~{augmentations_per_image} augmentations per original image")
    augmented_img_dir, augmented_mask_dir = create_directories(base_dir, component_name)
    print("Copying original files...")
    for img_file, mask_file in tqdm(component_files):
        img_path = os.path.join(base_dir, 'images', img_file)
        mask_path = os.path.join(base_dir, 'segmented_images', mask_file)
        shutil.copy(img_path, os.path.join(augmented_img_dir, img_file))
        shutil.copy(mask_path, os.path.join(augmented_mask_dir, mask_file))
    all_augmentations = [
        'horizontal_flip', 'vertical_flip', 
        'rotation_90', 'rotation_180', 'rotation_270',
        'brightness', 'contrast', 
        'small_rotation', 'translation', 'scale', 'shear'
    ]
    print("Generating augmented images...")
    augmentation_count = 0
    image_augmentation_history = {img_file: set() for img_file, _ in component_files}
    while augmentation_count < num_to_generate:
        random.shuffle(component_files)
        for img_file, mask_file in tqdm(component_files):
            if augmentation_count >= num_to_generate:
                break
            img_path = os.path.join(base_dir, 'images', img_file)
            mask_path = os.path.join(base_dir, 'segmented_images', mask_file)
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                print(f"Warning: Could not read {img_path} or {mask_path}")
                continue
            available_augs = [aug for aug in all_augmentations if aug not in image_augmentation_history[img_file]]
            if not available_augs:
                continue
            num_augs = min(augmentations_per_image, num_to_generate - augmentation_count, len(available_augs))
            if num_augs <= 0:
                break
            aug_types = random.sample(available_augs, num_augs)
            image_augmentation_history[img_file].update(aug_types)
            augmented_pairs = apply_augmentations(img, mask, aug_types)
            for aug_img, aug_mask, aug_type in augmented_pairs:
                if augmentation_count >= num_to_generate:
                    break
                orig_pixels = np.count_nonzero(mask)
                new_pixels = np.count_nonzero(aug_mask)
                if new_pixels < min_component_ratio * orig_pixels:
                    print(f"Warning: Augmentation {aug_type} lost too many pixels (ratio: {new_pixels/orig_pixels:.2f}). Falling back.")
                    aug_img, aug_mask, aug_type = safe_fallback_transform(img, mask, min_component_ratio)
                    # Record the fallback transform to help avoid exact duplicates.
                    image_augmentation_history[img_file].add(aug_type)
                base_name = Path(img_file).stem
                aug_img_file = f"{base_name}_aug_{aug_type}_{augmentation_count}.jpg"
                aug_mask_file = f"{base_name}_aug_{aug_type}_{augmentation_count}_semantic_mask.png"
                cv2.imwrite(os.path.join(augmented_img_dir, aug_img_file), aug_img)
                cv2.imwrite(os.path.join(augmented_mask_dir, aug_mask_file), aug_mask)
                augmentation_count += 1
                if augmentation_count >= num_to_generate:
                    break
    final_count = len(os.listdir(augmented_img_dir))
    print(f"Augmentation complete. Final count: {final_count} (Target: {target_count})")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    component_name = 'bolt3'
    target_count = 600
    augment_component_data(base_dir, component_name, target_count)
