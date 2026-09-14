# PicaSim Real-Time Horizon Detection & Attitude Estimation

## Title & Overview
This project implements a high-performance computer vision pipeline for real-time horizon detection and attitude estimation within the PicaSim flight simulator. While early iterations utilized zero-shot semantic segmentation via heavyweight neural networks to isolate the sky and ground, the final architecture pivots to a highly optimized, vectorized statistical variance algorithm. This engineering transition ensures ultra-low latency and a minimal memory footprint, making the core algorithmic logic viable for future real-world UAV deployment on resource-constrained microcontrollers.

## Tech Stack
* **Languages:** Python, C++ (via OpenCV backend)
* **Computer Vision:** OpenCV (cv2), NumPy
* **Deep Learning (Prototyping):** PyTorch, HuggingFace Transformers (SegFormer, CLIP), Segment Anything Model (SAM)
* **Screen Capture & Window Management:** MSS, PyGetWindow

## Key Features
* **Vectorized Variance Optimization:** Replaces iterative loops with multidimensional array broadcasting to simultaneously evaluate hundreds of candidate horizon lines by maximizing inter-class variance in the image's red channel.
* **Intelligent Lazy Caching:** Precomputes 3D coordinate grids and trigonometric masks upon initialization, eliminating redundant matrix calculations during the high-speed search loop.
* **Zero-Shot Segmentation Benchmarking:** Includes research implementations using SAM, CLIP, and SegFormer to generate pseudo-labels and ground-truth semantic masks for comparative analysis and dataset generation.
* **Dynamic Resolution Scaling:** Executes heavy mathematical operations on drastically downsampled image arrays before projecting high-fidelity telemetry overlays using nearest-neighbor interpolation.
* **Real-Time Attitude Conversion:** Mathematically translates the optimal visual boundary directly into actionable aircraft pitch and roll angles based on the camera's vertical field-of-view parameters.

## Hardware Requirements (If applicable)
* **Simulation Host:** Standard PC running Windows (required for PyGetWindow window targeting).

## Installation & Setup
1. Clone the repository and navigate to the project directory.
2. Create and activate a Python virtual environment.
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
3. Install the required dependencies.
```bash
pip install opencv-python numpy mss pygetwindow torch torchvision transformers segment-anything pillow
```
4. Launch the PicaSim flight simulator.
5. Execute the real-time screen tracking script to begin processing the telemetry.
```bash
python screen-recorder.py
```

## Project Architecture
* `find_horizon.py`: The core optimization engine. Contains the highly vectorized `find_horizon_vectorized` function for real-time processing and `calculate_attitude` for converting pixel boundaries to angles.
* `screen-recorder.py`: Handles dynamic simulator window localization, multithreaded screen capture using MSS, and renders the telemetry overlay at runtime.
* `variance_based_optimisation.py`: The foundational implementation of the variance-based search loop, serving as an algorithmic baseline before multidimensional vectorization was applied.
* `0_shot_sam_clip.py` & `0_shot_segformer.py`: Prototyping scripts utilizing pre-trained foundation models to generate high-accuracy semantic segmentation masks for training and validation.
* `split-video.py`: A data engineering script that processes raw `.mp4` flight footage into discrete frames at specific intervals for offline algorithmic testing.
