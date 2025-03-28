import os
import re
import cv2
import numpy as np
from collections import defaultdict

def combine_masks_and_split(input_dir, output_dir):
    """
    Combine component masks into single semantic segmentation masks and split them into folders.
    
    Args:
        input_dir (str): Directory containing the individual component masks.
        output_dir (str): Base directory where combined masks will be saved and split into folders.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define component classes and their corresponding pixel values
    component_classes = {
        'background': 0,
        'bolt': 1,
        'bolt washer': 2,
        'busbar': 3,
        'cable': 4,
        'connector': 5,
        'nut': 6,
        'plastic film': 7,
        'plastic cover': 8
    }
    
    # Create folders for each component and a "multi" folder
    component_folders = {component: os.path.join(output_dir, component.replace(" ", "_")) for component in component_classes.keys()}
    multi_folder = os.path.join(output_dir, "multi")
    
    for folder in component_folders.values():
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    if not os.path.exists(multi_folder):
        os.makedirs(multi_folder)
    
    # Updated regex pattern to capture labels with spaces and versions
    pattern = r'task-(\d+)-annotation-\d+-by-\d+-tag-([a-z ]+) - \d+'

    # Group files by task number
    task_files = defaultdict(list)
    for filename in os.listdir(input_dir):
        if not filename.endswith('.png'):
            continue
        
        match = re.search(pattern, filename)
        if match:
            task_id = match.group(1)
            component = match.group(2).strip()  # Extract label and remove extra spaces
            task_files[task_id].append((component, filename))
    
    # Process each task
    for task_id, files in task_files.items():
        if not files:
            continue
        
        # Get dimensions from first file
        first_file = os.path.join(input_dir, files[0][1])
        sample_mask = cv2.imread(first_file, cv2.IMREAD_GRAYSCALE)
        if sample_mask is None:
            print(f"Error reading file: {first_file}")
            continue
        
        height, width = sample_mask.shape
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        
        unique_components = set()
        
        for component, filename in files:
            # Strip any trailing version numbers from the label (e.g., "busbar - 3" -> "busbar")
            component_base = component.split(' - ')[0].strip()
            
            if component_base in component_classes:
                mask_path = os.path.join(input_dir, filename)
                component_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if component_mask is None:
                    print(f"Error reading file: {mask_path}")
                    continue
                
                # Normalize mask to binary (0 or 1)
                component_mask = (component_mask > 0).astype(np.uint8)
                
                # Add component to combined mask with class value
                combined_mask[component_mask > 0] = component_classes[component_base]
                unique_components.add(component_base)
            else:
                print(f"Unknown label '{component_base}' in file: {filename}")
        
        # Save combined mask to the appropriate folder
        output_filename = f"task-{task_id}-combined-mask.png"
        
        if len(unique_components) == 1:
            # Single-component mask
            single_component_name = list(unique_components)[0]
            output_path = os.path.join(component_folders[single_component_name], output_filename)
            cv2.imwrite(output_path, combined_mask)
            print(f"Saved {filename} to {component_folders[single_component_name]}")
        
        elif len(unique_components) > 1:
            # Multi-component mask
            output_path = os.path.join(multi_folder, output_filename)
            cv2.imwrite(output_path, combined_mask)
            print(f"Saved {filename} to {multi_folder}")
    
def create_colormap_visualization(component_classes, output_dir):
    """
    Create a colormap visualization image to help interpret the combined masks.
    
    Args:
        component_classes (dict): Dictionary mapping component names to pixel values.
        output_dir (str): Directory where visualization will be saved.
    """
    colors = {
        0: (0, 0, 0),       # Background (black)
        1: (255, 0, 0),     # Bolt (blue)
        2: (0, 255, 0),     # Boltwasher (green)
        3: (0, 0, 255),     # Busbar (red)
        4: (255, 255, 0),   # Cable (cyan)
        5: (255, 0, 255),   # Connector (magenta)
        6: (0, 255, 255),   # Nut (yellow)
        7: (128, 128, 128), # Plastic film (gray)
        8: (128, 0, 128)    # Plastic cover (purple)
    }
    
    height = len(component_classes) * 50
    width = 300
    colormap = np.zeros((height, width, 3), dtype=np.uint8)
    
    for component_name, value in component_classes.items():
        y_start = value * 50
        color = colors[value]
        
        cv2.rectangle(colormap, (0, y_start), (width - 1, y_start + 50), color[::-1], -1) 
        cv2.putText(colormap,
                    f"{value}: {component_name.title()}",
                    (10, y_start + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7,
                    color=(255,) *3)

# Example usage:
if __name__ == "__main__":
    input_directory = "labelstudiomasks"
    output_directory = "output"
    
    combine_masks_and_split(input_directory, output_directory)

