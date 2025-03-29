"""
This script extracts frames at 15-frame intervals
"""

import cv2
import os
import glob

def extract_frames(video_path, output_folder, frame_interval=15):
    """Extracts frames from a video at a specified interval and saves them in a folder named after the video."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]  # Get video name without extension
    save_path = os.path.join(output_folder, video_name)
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if video ends
        
        if frame_idx % frame_interval == 0:  # Extract frame every N frames
            img_path = os.path.join(save_path, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(img_path, frame)
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"Processed {video_name}: {saved_count} frames saved in {save_path}")

# Get all MP4 files in the script's folder
video_folder = os.getcwd()  # Get the current directory
output_dir = os.path.join(video_folder, "output_frames")  # Parent folder for all extracted frames
os.makedirs(output_dir, exist_ok=True)

# Define frame extraction interval for 60 FPS (e.g., every 15 frames = 4 FPS output)
frame_interval = 15  

# Process all MP4 videos in the folder
video_files = glob.glob(os.path.join(video_folder, "*.mp4"))
if not video_files:
    print("No MP4 files found in the folder.")

for video_file in video_files:
    extract_frames(video_file, output_dir, frame_interval)
