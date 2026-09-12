import os
import time

import cv2
import numpy as np

def find_horizon_variance():
    # Downsample the frame to reduce computation time
    scale_factor = 16
    downsampled_frame = cv2.resize(frame, (frame.shape[1] // scale_factor, frame.shape[0] // scale_factor))

    # Convert to grayscale
    # Using 32-bit float to prevent overflow or truncation errors later
    gray_frame = cv2.cvtColor(downsampled_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    frame_height, frame_width = gray_frame.shape

    # Create the coordinate grid to search through
    y, x = np.indices((frame_height, frame_width))
    # Shift the origin to the center of the frame
    x -= frame_width // 2
    y -= frame_height // 2

    # Define the candidate lines (pitch and roll search space)
    rolls = np.linspace(-np.pi, np.pi, 36) # Every 10 degrees
    pitches = np.linspace(-frame_height//2, frame_height//2, 20) # Every 5 degrees

    # Calculate the variance for each candidate line
    best_score = float('inf') # We are looking for the min variance
    best_line = None

    for roll in rolls:
        cos_r = np.cos(roll)
        sin_r = np.sin(roll)

        for pitch in pitches:
            # The perpendicular signed distance of any point (x, y)
            # from a line defined by normal angle theta and offset d is:
            # D = xcos(theta) + ysin(theta) - d
            # therefore x*cos(roll_angle) + y*sin(roll_angle) - pitch > 0
            mask = (x * cos_r + y * sin_r - pitch) > 0

            region_A = gray_frame[mask] #1D array of pixels where mask is True
            region_B = gray_frame[~mask] #1D array of pixels where mask is False

            # Skip invalid lines that are off the screen
            if len(region_A) == 0 or len(region_B) == 0:
                continue

            # Calculate weight (proportion of pixels)
            weight_A = len(region_A) / (frame_height * frame_width)
            weight_B = len(region_B) / (frame_height * frame_width)

            # Calculate the variance of each region
            variance_A = np.var(region_A)
            variance_B = np.var(region_B)

            intra_class_variance = (weight_A * variance_A) + (weight_B * variance_B)
            if intra_class_variance < best_score:
                best_score = intra_class_variance
                best_line = (roll, pitch)

    return best_line, scale_factor


def draw_horizon_line(best_line, scale_factor):
    roll = best_line[0]
    pitch = best_line[1]
    original_pitch = pitch * scale_factor

    # Find the coords of the closest point on the line to the center of the frame
    frame_height, frame_width = frame.shape[:2]
    cx = frame_width // 2
    cy = frame_height // 2
    cos_r = np.cos(roll)
    sin_r = np.sin(roll)
    # (x0, y0) is the closest point on the horizon line relative to centre
    x0 = original_pitch * cos_r
    y0 = original_pitch * sin_r

    # Project the line to onto the frame
    length = max(frame_height, frame_width) * 2 # Guaranteed to be longer than the frame diagonal
    # Point 1: anchor point + (direction * length)
    x1 = int(cx + x0 + length * (-sin_r))
    y1 = int(cy + y0 + length * cos_r)
    # Point 2: Anchor point - (direction * length)
    x2 = int(cx + x0 - length * (-sin_r))
    y2 = int(cy + y0 - length * cos_r)

    # Draw the line on the frame
    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return frame



images_dir = "trainingData/images"
output_dir = "trainingData/horizon_lines"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(images_dir):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    print(f"Processing {filename}...")
    frame = cv2.imread(os.path.join(images_dir, filename))

    best_line, scale_factor = find_horizon_variance()
    cv2.imwrite(os.path.join(output_dir, f"horizon_line_{filename}"), draw_horizon_line(best_line, scale_factor))





