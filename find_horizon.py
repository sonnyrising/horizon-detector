import os
import time

import cv2
import numpy as np
# Create a global cache to store the coordinate grids and trig values
CACHE = {}


def find_horizon_unified(frame):
    scale_factor = 16
    tiny_w = frame.shape[1] // scale_factor
    tiny_h = frame.shape[0] // scale_factor

    # 1. LAZY CACHING: Only calculate the heavy math grids once!
    # If the window size changes (or on the first frame), regenerate the cache.
    if CACHE.get('shape') != (tiny_h, tiny_w):
        y, x = np.indices((tiny_h, tiny_w))
        CACHE['x'] = x - tiny_w // 2
        CACHE['y'] = y - tiny_h // 2
        CACHE['rolls'] = np.linspace(-np.pi, np.pi, 36)
        CACHE['pitches'] = np.linspace(-tiny_h // 2, tiny_h // 2, 20)
        CACHE['cos'] = np.cos(CACHE['rolls'])
        CACHE['sin'] = np.sin(CACHE['rolls'])
        CACHE['shape'] = (tiny_h, tiny_w)

    # 2. Downsample and extract the Red Channel
    downsampled = cv2.resize(frame, (tiny_w, tiny_h))
    gray = downsampled[:, :, 2].astype(np.float32)

    best_score = -1
    best_line = None
    THRESHOLD = 200
    total_pixels = tiny_h * tiny_w

    # 3. The high-speed search loop
    for idx, roll in enumerate(CACHE['rolls']):
        cos_r = CACHE['cos'][idx]
        sin_r = CACHE['sin'][idx]

        for pitch in CACHE['pitches']:
            # Create the boolean mask using cached grids
            mask = (CACHE['x'] * cos_r + CACHE['y'] * sin_r - pitch) > 0

            # Count pixels without allocating new arrays
            count_A = np.count_nonzero(mask)
            count_B = total_pixels - count_A

            if count_A == 0 or count_B == 0:
                continue

            # 4. THE SPEED TRICK: np.sum instead of array slicing
            mu_A = np.sum(gray, where=mask) / count_A
            mu_B = np.sum(gray, where=~mask) / count_B

            w_A = count_A / total_pixels
            w_B = count_B / total_pixels

            # Maximize inter-class variance
            inter_var = w_A * w_B * (mu_A - mu_B) ** 2

            if inter_var > best_score:
                best_score = inter_var
                best_line = (roll, pitch)

    if best_score < THRESHOLD:
        return None, scale_factor

    return best_line, scale_factor


def draw_horizon_line(best_line, scale_factor, frame):
    roll = best_line[0]
    pitch = best_line[1]
    original_pitch = pitch * scale_factor

    frame_height, frame_width = frame.shape[:2]
    cx = frame_width // 2
    cy = frame_height // 2
    cos_r = np.cos(roll)
    sin_r = np.sin(roll)

    x0 = original_pitch * cos_r
    y0 = original_pitch * sin_r

    # 1. Work entirely at the TINY resolution to save memory bandwidth
    tiny_h = frame_height // scale_factor
    tiny_w = frame_width // scale_factor

    # Re-use the cached coordinate grid if available, otherwise generate it
    if 'x' in CACHE and CACHE['shape'] == (tiny_h, tiny_w):
        tiny_x, tiny_y = CACHE['x'], CACHE['y']
    else:
        tiny_y, tiny_x = np.indices((tiny_h, tiny_w))
        tiny_x = tiny_x - (tiny_w // 2)
        tiny_y = tiny_y - (tiny_h // 2)

    tiny_mask = (tiny_x * cos_r + tiny_y * sin_r - pitch) > 0

    # 2. Check brightness on the TINY image (Instantaneous)
    tiny_gray = cv2.resize(frame, (tiny_w, tiny_h))
    tiny_gray = cv2.cvtColor(tiny_gray, cv2.COLOR_BGR2GRAY)

    # Use np.sum to avoid allocating new arrays during the check
    count_A = np.count_nonzero(tiny_mask)
    count_B = (tiny_h * tiny_w) - count_A

    # Prevent division by zero if the line is completely off-screen
    if count_A == 0 or count_B == 0:
        return frame

    mean_A = np.sum(tiny_gray, where=tiny_mask) / count_A
    mean_B = np.sum(tiny_gray, where=~tiny_mask) / count_B

    orange_bgr = (0, 165, 255)
    blue_bgr = (255, 0, 0)

    # 3. Create a TINY colored image (Instantaneous)
    tiny_overlay = np.zeros((tiny_h, tiny_w, 3), dtype=np.uint8)

    if mean_A > mean_B:
        tiny_overlay[tiny_mask] = blue_bgr
        tiny_overlay[~tiny_mask] = orange_bgr
    else:
        tiny_overlay[tiny_mask] = orange_bgr
        tiny_overlay[~tiny_mask] = blue_bgr

    # 4. Stretch the fully colored tiny image to 1080p using C++ backend
    full_overlay = cv2.resize(tiny_overlay, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

    # 5. Blend it with the main frame
    alpha = 0.5
    cv2.addWeighted(full_overlay, alpha, frame, 1 - alpha, 0, frame)

    # 6. Draw the crisp, full-resolution line on top
    length = max(frame_height, frame_width) * 2
    x1 = int(cx + x0 + length * (-sin_r))
    y1 = int(cy + y0 + length * cos_r)
    x2 = int(cx + x0 - length * (-sin_r))
    y2 = int(cy + y0 - length * cos_r)

    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return frame


def find_horizon_vectorized(frame):
    scale_factor = 16
    tiny_w = frame.shape[1] // scale_factor
    tiny_h = frame.shape[0] // scale_factor

    # 1. LAZY CACHING: Precompute all 720 masks as a single 3D block
    if CACHE.get('shape') != (tiny_h, tiny_w):
        y, x = np.indices((tiny_h, tiny_w))
        x = x - tiny_w // 2
        y = y - tiny_h // 2

        rolls = np.linspace(-np.pi, np.pi, 36)
        pitches = np.linspace(-tiny_h // 2, tiny_h // 2, 20)

        # Create every combination of roll and pitch
        roll_grid, pitch_grid = np.meshgrid(rolls, pitches, indexing='ij')
        roll_flat = roll_grid.flatten()
        pitch_flat = pitch_grid.flatten()

        CACHE['params'] = list(zip(roll_flat, pitch_flat))

        # Reshape for 3D broadcasting: (720, 1, 1)
        cos_r = np.cos(roll_flat).reshape(-1, 1, 1)
        sin_r = np.sin(roll_flat).reshape(-1, 1, 1)
        p = pitch_flat.reshape(-1, 1, 1)

        # Instantly calculate all 720 masks at once. Shape: (720, Height, Width)
        masks = (x * cos_r + y * sin_r - p) > 0

        # Precalculate sizes and weights
        count_A = np.count_nonzero(masks, axis=(1, 2))
        total_pixels = tiny_h * tiny_w

        # Filter out invalid lines (completely off-screen)
        valid = (count_A > 0) & (count_A < total_pixels)
        CACHE['valid_params'] = [CACHE['params'][i] for i in range(len(valid)) if valid[i]]

        CACHE['masks_valid'] = masks[valid]
        CACHE['count_A'] = count_A[valid]
        CACHE['count_B'] = total_pixels - CACHE['count_A']
        CACHE['w_A'] = CACHE['count_A'] / total_pixels
        CACHE['w_B'] = CACHE['count_B'] / total_pixels
        CACHE['shape'] = (tiny_h, tiny_w)

    # 2. REAL-TIME LOOP (No Python 'for' loops!)
    downsampled = cv2.resize(frame, (tiny_w, tiny_h))
    gray = downsampled[:, :, 2].astype(np.float32)

    # Apply all 720 masks to the image simultaneously
    sum_A = np.sum(gray * CACHE['masks_valid'], axis=(1, 2))
    sum_B = np.sum(gray) - sum_A

    # Calculate all 720 scores at once using array mathematics
    mu_A = sum_A / CACHE['count_A']
    mu_B = sum_B / CACHE['count_B']
    inter_var = CACHE['w_A'] * CACHE['w_B'] * (mu_A - mu_B) ** 2

    # Find the absolute best score instantly
    best_idx = np.argmax(inter_var)
    best_score = inter_var[best_idx]

    if best_score < 200:
        return None, scale_factor

    return CACHE['valid_params'][best_idx], scale_factor


def calculate_attitude(best_line, scale_factor, frame_height, v_fov=60.0):
    """
    Converts the mathematical line parameters into aircraft attitude angles.
    v_fov is the Vertical Field of View of the simulator camera in degrees.
    """
    if best_line is None:
        return (None, None)

    roll_rad, pitch_pixels = best_line

    # Convert roll to degrees
    roll_angle = np.degrees(roll_rad)

    # Convert pixel distance to pitch angle
    # Scale the downsampled pixel distance back to the full resolution
    real_pitch_pixels = pitch_pixels * scale_factor
    # Find the maximum possible pixel distance from the center
    max_pitch_pixels = frame_height / 2.0
    pitch_ratio = real_pitch_pixels / max_pitch_pixels
    # Map that ratio to the camera's field of view
    pitch_angle = pitch_ratio * (v_fov / 2.0)

    return (roll_angle, pitch_angle)


