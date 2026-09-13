import numpy as np
import cv2
import time
import mss
import find_horizon
import pygetwindow as gw

# Dynamically find the coords of the sim window
windows = gw.getWindowsWithTitle("PicaSim")
if not windows:
    print("PicaSim window not found.")
    exit()

picasim_window = windows[0]
bounding_box = {
    "top": int(picasim_window.top + 80),
    "left": int(picasim_window.left + 50),
    "width": int(picasim_window.width * 1.5),
    "height": int(picasim_window.height * 2)
}

horizon_finder = find_horizon

#Initialise MSS
with mss.MSS() as sct:
    while True:
        start_time = time.time()
        screenshot = sct.grab(bounding_box)

        frame_bgra = np.array(screenshot)
        frame = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

        best_line, scale_factor = horizon_finder.find_horizon_vectorized(frame)

        if best_line is not None:
            cv2.imshow("PicaSim Autopilot Vision", horizon_finder.draw_horizon_line(best_line, scale_factor, frame))
        else:
            cv2.imshow("PicaSim Autopilot Vision", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        fps = 1 / (time.time() - start_time)
        print(f"FPS: {fps:.2f}")
    cv2.destroyAllWindows()