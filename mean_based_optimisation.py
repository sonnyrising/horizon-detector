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
    best_score = -1 # We are looking for the max
    best_line = None

    total_pixels = frame_height * frame_width

    for roll in rolls:
        cos_r = np.cos(roll)
        sin_r = np.sin(roll)

        for pitch in pitches:
            mask = (x * cos_r + y * sin_r - pitch) > 0

            region_A = gray_frame[mask]
            region_B = gray_frame[~mask]

            if len(region_A) == 0 or len(region_B) == 0:
                continue

            w_A = len(region_A) / total_pixels
            w_B = len(region_B) / total_pixels

            # Calculate Averages (Means)
            mu_A = np.mean(region_A)
            mu_B = np.mean(region_B)

            # Calculate Noise (Variances)
            var_A = np.var(region_A)
            var_B = np.var(region_B)

            # 1. How far apart are the averages?
            inter_var = w_A * w_B * (mu_A - mu_B)**2

            # 2. How noisy are the regions internally?
            intra_var = (w_A * var_A) + (w_B * var_B)

            # Prevent division by zero if a region is perfectly flat
            if intra_var == 0:
                intra_var = 1e-5

            # 3. Fisher's Ratio: Maximize separation, minimize noise
            fisher_score = inter_var / intra_var

            if fisher_score > best_score:
                best_score = fisher_score
                best_line = (roll, pitch)

    return best_line, scale_factor

def find_horizon_red():
    # Downsample the frame to reduce computation time
    scale_factor = 16
    downsampled_frame = cv2.resize(frame, (frame.shape[1] // scale_factor, frame.shape[0] // scale_factor))

    #Use the Red channel to force the Sea and Land to group together
    # OpenCV uses BGR, so index 2 is the Red channel
    gray = downsampled_frame[:, :, 2].astype(np.float32)

    height, width = gray.shape
    y, x = np.indices((height, width))
    x = x - width // 2
    y = y- height // 2

    rolls = np.linspace(-np.pi, np.pi, 36)
    pitches = np.linspace(-height//2, height//2, 20)

    best_score = -1 # we want to maximise the score
    best_line = None

    # The threshold for the score to be considered a valid line
    THRESHOLD = 200

    total_pixels = height * width

    for roll in rolls:
        cos_r = np.cos(roll)
        sin_r = np.sin(roll)

        for pitch in pitches:
            mask = (x * cos_r + y * sin_r - pitch) > 0
            region_A = gray[mask]
            region_B = gray[~mask]

            if len(region_A) == 0 or len(region_B) == 0:
                continue

            weight_A = len(region_A) / total_pixels
            weight_B = len(region_B) / total_pixels

            # Find the average brightness of each region
            mu_A = np.mean(region_A)
            mu_B = np.mean(region_B)

            # Calculate variance
            inter_class_variance = weight_A * weight_B * (mu_A - mu_B)**2

            if inter_class_variance > best_score:
                best_score = inter_class_variance
                best_line = (roll, pitch)

    if best_score < THRESHOLD:
                return None, scale_factor

    return best_line, scale_factor

def find_horizon_unified():
    # Downsample the frame to reduce computation time
    scale_factor = 16
    downsampled_frame = cv2.resize(frame, (frame.shape[1] // scale_factor, frame.shape[0] // scale_factor))

    # Use the brightness channel to seperate bright sky from dark land and sea
    hsv = cv2.cvtColor(downsampled_frame, cv2.COLOR_BGR2HSV)
    gray = hsv[:, :, 2].astype(np.float32) #? what does [:,:,2] mean?

    height, width = gray.shape
    y, x = np.indices((height, width))
    x = x - width // 2
    y = y- height // 2

    rolls = np.linspace(-np.pi, np.pi, 36)
    pitches = np.linspace(-height//2, height//2, 20)

    best_score = -1 # we want to maximise the score
    best_line = None

    # The threshold for the score to be considered a valid line
    THRESHOLD = 500

    total_pixels = height * width

    for roll in rolls:
        cos_r = np.cos(roll)
        sin_r = np.sin(roll)

        for pitch in pitches:
            mask = (x * cos_r + y * sin_r - pitch) > 0
            region_A = gray[mask]
            region_B = gray[~mask]

            if len(region_A) == 0 or len(region_B) == 0:
                continue

            weight_A = len(region_A) / total_pixels
            weight_B = len(region_B) / total_pixels

            # Find the average brightness of each region
            mu_A = np.mean(region_A)
            mu_B = np.mean(region_B)

            # Calculate variance
            inter_class_variance = weight_A * weight_B * (mu_A - mu_B)**2

            if inter_class_variance > best_score:
                best_score = inter_class_variance
                best_line = (roll, pitch)

    if best_score < THRESHOLD:
                return None, scale_factor

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
output_dir = "trainingData/horizon_lines_fisher"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(images_dir):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    print(f"Processing {filename}...")
    frame = cv2.imread(os.path.join(images_dir, filename))

    best_line, scale_factor = find_horizon_variance()

    # Only draw and save if a valid horizon was found
    if best_line is not None:
        drawn_frame = draw_horizon_line(best_line, scale_factor)
        cv2.imwrite(os.path.join(output_dir, f"horizon_line_{filename}"), drawn_frame)
    else:
        print(f"Skipping {filename}: No confident horizon detected.")
        cv2.imwrite(os.path.join(output_dir, f"horizon_line_{filename}"), frame)