import cv2
import os

def extract_frames(video_path, output_folder, frame_interval=10):
    """
    Extract frames from a video at specified intervals.
    
    Args:
        video_path (str): Path to the input video file
        output_folder (str): Path to save extracted frames
        frame_interval (int): Extract every nth frame
    """
    # Create video-specific output folder using video filename
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_folder = os.path.join(output_folder, video_name)
    
    if not os.path.exists(video_output_folder):
        os.makedirs(video_output_folder)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return 0
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nProcessing video: {video_name}")
    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Extracting every {frame_interval}th frame...")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        success, frame = video.read()
        
        if not success:
            break
            
        if frame_count % frame_interval == 0:
            # Save frame as image
            output_path = os.path.join(video_output_folder, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
            
        frame_count += 1
        
        # Optional: Print progress
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")
    
    video.release()
    print(f"Saved {saved_count} frames from {video_name}")
    return saved_count

def process_video_directory(input_directory, output_directory, frame_interval=10):
    """
    Process all MOV files in the input directory.
    
    Args:
        input_directory (str): Directory containing MOV files
        output_directory (str): Directory to save extracted frames
        frame_interval (int): Extract every nth frame
    """
    # Create main output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Get all MOV files in the directory
    video_files = [f for f in os.listdir(input_directory) if f.lower().endswith('.mov')]
    
    if not video_files:
        print("No MOV files found in the input directory!")
        return
    
    print(f"Found {len(video_files)} MOV files")
    total_frames_saved = 0
    
    # Process each video
    for video_file in video_files:
        video_path = os.path.join(input_directory, video_file)
        frames_saved = extract_frames(video_path, output_directory, frame_interval)
        total_frames_saved += frames_saved
    
    print(f"\nExtraction complete!")
    print(f"Total frames saved across all videos: {total_frames_saved}")

# Example usage
if __name__ == "__main__":
    input_directory = "UnsortedVideos"  # Your input directory with MOV files
    output_directory = "ExtractedFrames"  # Where to save the frames
    process_video_directory(input_directory, output_directory, frame_interval=10)