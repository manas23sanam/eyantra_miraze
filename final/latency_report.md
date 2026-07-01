# Latency Research Report: 2D-to-3D Skeleton Tracking Performance Analysis

This report provides a comparative performance analysis of running 2D-to-3D skeletal tracking for physical therapy exercise monitoring. It compares the performance on a high-specification development laptop against the projected performance on an edge platform like the Raspberry Pi 4.

---

## 1. Latency Benchmark Comparison

The tracking pipeline's performance was evaluated under two configurations:

### A. Baseline Configuration (Empty Placeholder)
- **Source**: `latency_results_summary.csv`
- **Average Latency**: **~0.25 ms**
- **Peak Latency (p95)**: **~0.35 ms**
- **Average Frame Rate**: **> 3800 FPS**
- **Analysis**: This configuration uses a dummy/empty placeholder for the detector where frame data is passed but no actual model inference or depth-map alignment is performed. The sub-millisecond latency represents the absolute floor of Python program overhead and camera frame ingestion without processing.

### B. Production Configuration (MediaPipe + 3D RealSense Geometry)
- **Source**: `squat_angle_validation_results.csv`
- **Average Latency**: **~55 ms to 70 ms** (mean production latency: **~60 ms**)
- **Peak Latency (p95)**: **~65 ms to 108 ms**
- **Average Frame Rate**: **14 to 18 FPS**
- **Analysis**: This configuration runs the complete physical therapy pipeline, which includes color-to-depth RealSense alignment, MediaPipe Pose landmark extraction, and real-world 3D coordinate projection. This represents a heavy compute workload with real-time feedback constraints.

---

## 2. Computational Bottlenecks

An end-to-end frame processing cycle consists of three primary stages:

1. **RealSense Depth-to-Color Alignment (~5–15 ms)**
   - To associate depth readings with RGB pixels, the high-resolution depth map is spatially warped and mapped to the color camera's coordinate frame.
   - This requires per-pixel matrix transformations and interpolation, creating a constant CPU-intensive overhead.

2. **MediaPipe Pose Inference (~30–45 ms)**
   - The BlazePose convolutional neural network runs on the CPU to detect 33 landmarks.
   - Forward pass execution of deep neural networks on general-purpose CPU threads constitutes the single largest latency component.

3. **3D Coordinate Projection & Trigonometric Math (~2–5 ms)**
   - Pixels corresponding to key landmarks are projected into 3D camera space using camera intrinsics ($f_x$, $f_y$, $c_x$, $c_y$) and depth values:
     $$X = \frac{(x - c_x) \cdot Z}{f_x}, \quad Y = \frac{(y - c_y) \cdot Z}{f_y}$$
   - Vectors are constructed and 3D angles are calculated using dot products. While mathematically simple, this operation scales with the number of tracked joints.

---

## 3. Expected Constraints on Raspberry Pi 4

Deploying this pipeline directly onto a Raspberry Pi 4 (Quad-core ARM Cortex-A72 @ 1.5 GHz, 4GB/8GB RAM) introduces severe constraints:

* **Thermal Throttling**: Continuous 100% CPU usage on all four cores during real-time image processing causes rapid heat accumulation. Without active cooling (e.g., heatsinks + fan), the CPU frequency throttles from 1.5 GHz down to 750 MHz within minutes, halving performance.
* **Lack of GPU Acceleration**: MediaPipe relies heavily on GPU/NPU acceleration. The Raspberry Pi 4's VideoCore VI GPU does not support CUDA or robust OpenGL/Vulkan compute pipelines for TensorFlow Lite, forcing the model to run purely on the CPU, utilizing only NEON SIMD instructions.
* **Low Frame Rate**: The expected frame rate drops to **3–8 FPS** (down from 15–18 FPS on the laptop), resulting in choppy rendering, delayed posture feedback, and missed reps.
* **High Tail Latency (p95 > 500 ms)**: Due to shared system memory, lower memory bandwidth (LPDDR4), and OS task-switching overhead, latency spikes occur frequently, degrading the user experience.

---

## 4. Recommended Optimization Strategies

To achieve a target of **30 FPS** and **< 33 ms latency** on resource-constrained edge hardware, the following optimizations are recommended:

1. **Input Resolution & Stream Downscaling**
   - Capture RealSense color and depth frames at `320x240` resolution rather than `640x480`. This reduces the alignment workload by **75%** (from 307,200 pixels to 76,800 pixels).

2. **Model Selection & Quantization**
   - Swap the default MediaPipe Pose model for the **BlazePose Lite** or **int8/float16 quantized TFLite** model. Quantized execution significantly speeds up execution on ARM NEON architectures.

3. **Asynchronous Multi-threaded Pipeline**
   - Decouple the frame capture, pose inference, and UI rendering into independent threads:
     - **Thread 1**: Ingests camera streams and performs depth alignment (runs at 30 Hz).
     - **Thread 2**: Runs model inference on the latest available frame (runs at 10–15 Hz).
     - **Thread 3**: Renders the skeleton UI, interpolating joint positions between inferences to maintain a smooth 30 FPS display.

4. **Dedicated Hardware Accelerators**
   - Integrate an **Edge TPU (Coral USB Accelerator)** or **Intel Neural Compute Stick 2 (OpenVINO)**. Offloading pose estimation to dedicated silicon drops inference latency to **< 10 ms**, freeing the Pi CPU for alignment and HUD logic.
