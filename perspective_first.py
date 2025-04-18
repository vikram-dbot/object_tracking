import cv2
import numpy as np
import os

def equirectangular_to_perspective(e_img, fov, theta, phi, height, width):
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    fov = np.deg2rad(fov)
    focal_length = 0.5 * width / np.tan(0.5 * fov)

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])

    R_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    R = R_x @ R_y

    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            x_ndc = (x - width / 2) / focal_length
            y_ndc = (y - height / 2) / focal_length
            vec = np.array([x_ndc, y_ndc, 1.0])
            vec = vec / np.linalg.norm(vec)
            vec = R @ vec
            lon = np.arctan2(vec[0], vec[2])
            lat = np.arcsin(vec[1])
            map_x[y, x] = (lon / np.pi + 1.0) * (e_img.shape[1] - 1) / 2
            map_y[y, x] = (lat / (0.5 * np.pi) + 1.0) * (e_img.shape[0] - 1) / 2

    return cv2.remap(e_img, map_x, map_y, cv2.INTER_LINEAR)

# === CONFIGURATION ===
video_folder = rf"C:\Users\Welcome\Desktop\final_testing"       
output_root = rf"C:\Users\Welcome\phases_detection\output\dta"
fov = 90 
view_angles = [0, 45, 90, 135, 180, 225, 270, 315] 
height = 512
width = 512

os.makedirs(output_root, exist_ok=True)

# === PROCESSING LOOP ===
for filename in os.listdir(video_folder):
    if filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
        video_path = os.path.join(video_folder, filename)
        video_name = os.path.splitext(filename)[0]
        output_folder = os.path.join(output_root, video_name)
        os.makedirs(output_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps // 5)  # For 2 frames per second

        frame_idx = 0
        saved_frame_idx = 0

        print(f"🎥 Processing video: {filename}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                for i, theta in enumerate(view_angles):
                    persp_view = equirectangular_to_perspective(frame, fov, theta, phi=0, height=height, width=width)
                    out_filename = os.path.join(output_folder, f"frame{saved_frame_idx:04d}_view{i}.jpg")
                    cv2.imwrite(out_filename, persp_view)
                print(f"✅ Frame {saved_frame_idx} processed with {len(view_angles)} views.")
                saved_frame_idx += 1

            frame_idx += 1

        cap.release()
        print(f"🎉 Done with {filename}! Output saved to: {output_folder}\n")

print("🚀 All videos processed.")
