import os
import shutil
from collections import defaultdict

def mirror_output_structure(mask_dir, analysis_file, upload_dir, organized_dir):
    """
    Mirror the output folder structure with original images matched by task ID
    
    Args:
        mask_dir (str): Path to folder with masks (output directory structure)
        analysis_file (str): Path to analysis file from previous step
        upload_dir (str): Directory with original uploaded images
        organized_dir (str): Target directory to mirror output structure
    """
    # Create mapping from analysis file
    task_map = {}
    with open(analysis_file, 'r') as f:
        current_task = {}
        for line in f:
            line = line.strip()
            if line.startswith("Task ID:"):
                current_task["id"] = line.split(": ")[1]
            elif line.startswith("Image:"):
                current_task["image"] = line.split(": ")[1]
            elif line == "" and current_task:
                task_map[current_task["id"]] = current_task["image"]
                current_task = {}

    # Build index of available images
    image_index = defaultdict(list)
    for root, _, files in os.walk(upload_dir):
        for file in files:
            image_index[file].append(os.path.join(root, file))

    # Process mask directory structure
    for root, dirs, files in os.walk(mask_dir):
        # Create mirror structure in organized_dir
        relative_path = os.path.relpath(root, mask_dir)
        organized_root = os.path.join(organized_dir, relative_path)
        os.makedirs(organized_root, exist_ok=True)

        # Process each mask file
        for file in files:
            if file.endswith("-combined-mask.png"):
                # Extract task ID from mask filename
                task_id = file.split("-")[1]
                
                # Get original image name from analysis mapping
                original_image = task_map.get(task_id)
                
                if original_image and original_image in image_index:
                    # Get first matching image path
                    src_path = image_index[original_image][0]
                    
                    # Create destination path
                    dest_path = os.path.join(organized_root, f"task-{task_id}.jpg")
                    
                    # Copy image
                    shutil.copy2(src_path, dest_path)
                    print(f"Copied {original_image} => {dest_path}")
                else:
                    print(f"Missing image for task {task_id}")

if __name__ == "__main__":
    # Configuration
    MASK_DIR = "output"              # Folder with mask files and structure
    ANALYSIS_FILE = "output_analysis.txt"
    UPLOAD_DIR = "upload"            # Original uploaded images
    ORGANIZED_DIR = "organized_images"  # Target directory

    # Create organized images directory
    shutil.rmtree(ORGANIZED_DIR, ignore_errors=True)
    os.makedirs(ORGANIZED_DIR, exist_ok=True)

    mirror_output_structure(MASK_DIR, ANALYSIS_FILE, UPLOAD_DIR, ORGANIZED_DIR)
    print("Folder structure mirroring complete!")
