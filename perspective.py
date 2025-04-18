import pandas as pd # type: ignore
import cv2 # type: ignore
import os
import glob
from omnicv import fisheyeImgConv # type: ignore
import numpy as np # type: ignore

# Fix deprecated NumPy aliases
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool

def extract_frames_from_video(video_path, output_folder, frame_interval=1):
    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video has {total_frames} frames at {fps} FPS")
    
    # Determine frame indices to extract (every frame_interval)
    frame_indices = range(0, total_frames, frame_interval)
    
    # Determine zero-padding to at least 4 digits
    num_digits = max(4, len(str(len(frame_indices) - 1)))
    
    # Process video and extract frames
    frame_count = 0
    for idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # Move to the frame
        ret, frame = cap.read()
        if not ret:
            print(f"Frame {frame_idx} not found!")
            continue
        
        # Save frame with zero-padded numbering
        frame_filename = os.path.join(output_folder, f"image_{str(idx).zfill(num_digits)}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
        print(f"Saved: {frame_filename}")
    
    # Cleanup
    cap.release()
    print(f"✅ {frame_count} frames extracted and saved to '{output_folder}'")
    return frame_count

def equi_to_pers(img_path, number_division, img_reso, output_path):
    mapper = fisheyeImgConv()

    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        return
        
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    h, w, _ = img.shape
    if (w/h) != 2:
        print(f"Input image {img_path} is not a valid equirectangular image (aspect ratio 2:1).")
        return 
        
    each_angle_variation = int(360/number_division)
    cube_faces = {}
    it = 45
    
    for i in range(number_division-1, -1, -1):
        FOV = 90
        Phi = 0
        persp = mapper.eqruirect2persp(img, FOV, it, Phi, img_reso, img_reso)
        cube_faces[i] = persp
        it = it + each_angle_variation

    for face_name, face_img in cube_faces.items():
        output_file = os.path.join(output_path, f"{base_name}_{face_name}.jpg")
        cv2.imwrite(output_file, face_img)
        print(f"Saved perspective view: {output_file}")

def process_folder(input_folder, number_of_split, output_folder, output_img_size):
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files in the folder
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = []
    for ext in supported_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(input_folder, f"*{ext.upper()}")))
    if not image_files:
        print(f"No supported image files found in {input_folder}.")
        return
    print(f"Found {len(image_files)} images in {input_folder}. Processing...")

    # Process each image
    for image_path in sorted(image_files):
        try:
            print(f"Processing: {image_path}")
            equi_to_pers(image_path, number_of_split, output_img_size, output_folder)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

def main():
    # Configuration parameters
    video_path = r"C:\Users\Welcome\Desktop\VID_20250417_160053_00_252.mp4"
    frame_output_folder = r"C:\Users\Welcome\phases_detection\output\video61"
    perspective_output_folder = r"C:\Users\Welcome\phases_detection\output\video61\perspective"   
    # Parameters for perspective conversion
    number_of_split = 8  # Number of perspective views to create (8 for 45° intervals)
    output_img_size = 2000  # Resolution for perspective images

    # Extract frames (every 30th frame for example - adjust as needed)
    frame_interval = 25  # Extract every 30th frame

    # Step 1: Extract frames from video
    print("Step 1: Extracting frames from video...")
    extract_frames_from_video(video_path, frame_output_folder, frame_interval)

    # Step 2: Convert extracted frames to perspective views
    print("\nStep 2: Converting equirectangular frames to perspective views...")
    process_folder(frame_output_folder, number_of_split, perspective_output_folder, output_img_size)
    print("\nProcess completed!")

if __name__ == "__main__":
    main()