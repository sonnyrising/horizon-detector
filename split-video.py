# Splits an mp4 file into multiple frames by taking every 30th frame

import os
import cv2

current_video = "video1"

output_dir = "trainingData"
os.makedirs(output_dir, exist_ok=True)

video_path = f"./trainingData/videos/{current_video}.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file at '{video_path}'. Check the path and extension.")

frame_interval = 30
count = 0
saved = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if count % frame_interval == 0:
        file_path = os.path.join(output_dir, f"{current_video}_{saved:04d}.jpg")
        success = cv2.imwrite(file_path, frame)
        if success:
            saved += 1
        else:
            print(f"Warning: Failed to save frame {saved}")

    count += 1

cap.release()
print(f"Done! Successfully saved {saved} frames to '{output_dir}'.")