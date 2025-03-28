import os
import cv2
import numpy as np

def visualize_masks_by_component(input_base_dir, output_base_dir, component_classes):
    """
    Create colored visualizations of combined masks and save them into subfolders based on components.
    
    Args:
        input_base_dir (str): Base directory containing folders for different components.
        output_base_dir (str): Base directory where visualizations will be saved.
        component_classes (dict): Dictionary mapping component names to pixel values.
    """
    # Define colors for each class (BGR format for OpenCV)
    colors = {
        0: (0, 0, 0),       # Background (black)
        1: (255, 0, 0),     # Bolt (blue)
        2: (0, 255, 0),     # Bolt Washer (green)
        3: (0, 0, 255),     # Busbar (red)
        4: (255, 255, 0),   # Cable (cyan)
        5: (255, 0, 255),   # Connector (magenta)
        6: (0, 255, 255),   # Nut (yellow)
        7: (128, 128, 128), # Plastic Film (gray)
        8: (128, 0, 128)    # Plastic Cover (purple)
    }
    
    # Ensure the output base directory exists
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    # Process each folder in the input base directory
    for folder_name in os.listdir(input_base_dir):
        folder_path = os.path.join(input_base_dir, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
        
        # Create corresponding output folder
        output_folder = os.path.join(output_base_dir, folder_name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        print(f"Processing folder: {folder_name}")
        
        # Process each mask file in the current folder
        for filename in os.listdir(folder_path):
            if filename.endswith('combined-mask.png'):
                mask_path = os.path.join(folder_path, filename)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if mask is None:
                    print(f"Error reading file: {mask_path}")
                    continue
                
                print(f"Unique values in mask {filename}: {np.unique(mask)}")
                
                # Create colored visualization
                visualization = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                
                # Map grayscale values to colors
                for class_idx in range(9):  # Assuming classes are from 0 to 8
                    if class_idx in colors:
                        visualization[mask == class_idx] = colors[class_idx]
                
                # Save visualization to the output folder
                vis_filename = filename.replace('combined-mask', 'visualization')
                output_path = os.path.join(output_folder, vis_filename)
                cv2.imwrite(output_path, visualization)
                print(f"Created visualization: {output_path}")

# Example usage:
if __name__ == "__main__":
    input_directory = "output"          # Base directory containing folders with combined masks
    output_directory = "visualisation" # Base directory to save visualizations
    
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
    
    visualize_masks_by_component(input_directory, output_directory, component_classes)
