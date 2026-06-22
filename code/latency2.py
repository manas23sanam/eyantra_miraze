import pyrealsense2 as rs
import time
import cv2
import numpy as np
import os
import glob
import csv

# ============================================================
# CONFIG: point this at the folder that contains your .db3 files
# ============================================================
DATASET_FOLDER = r"C:\Users\BIT\OneDrive\Desktop\software mirror\new"
OUTPUT_CSV = "latency_results_summary.csv"


def run_inference_placeholder(depth_frame):
    """
    PUT YOUR ACTUAL MODEL INFERENCE HERE.
    This function should take the depth_frame and return whatever
    your model returns (squat angle, posture classification, etc).
    Right now it does nothing, so the latency you measure is just
    a baseline / sanity check until you plug your real model in.
    """
    depth_image = np.asanyarray(depth_frame.get_data())
    # Example placeholder "work" so timing isn't literally zero:
    _ = cv2.convertScaleAbs(depth_image, alpha=0.03)
    return None


def process_single_file(file_path, show_window=False):
    """Runs one .db3 recording through the pipeline and returns latency stats."""
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, file_path, repeat_playback=False)

    latency_records = []
    frame_count = 0

    try:
        profile = pipeline.start(config)
        # Let the playback device run at its own recorded pace? For latency
        # benchmarking we usually want to go AS FAST AS POSSIBLE, not real-time.
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)  # <-- key line, explained below

        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError:
                # No more frames / end of file
                break

            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

            frame_count += 1

            # ================= LATENCY TIMER =================
            start_time = time.perf_counter()
            run_inference_placeholder(depth_frame)
            end_time = time.perf_counter()
            # ===================================================

            latency_ms = (end_time - start_time) * 1000
            latency_records.append(latency_ms)

            if show_window:
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
                )
                cv2.imshow('Playback', depth_colormap)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        pipeline.stop()
        if show_window:
            cv2.destroyAllWindows()

    if not latency_records:
        return None

    arr = np.array(latency_records)
    return {
        "file": os.path.basename(file_path),
        "frames": frame_count,
        "avg_latency_ms": float(np.mean(arr)),
        "min_latency_ms": float(np.min(arr)),
        "max_latency_ms": float(np.max(arr)),
        "p95_latency_ms": float(np.percentile(arr, 95)),
        "avg_fps": float(1000 / np.mean(arr)) if np.mean(arr) > 0 else 0,
    }


def main():
    db3_files = sorted(glob.glob(os.path.join(DATASET_FOLDER, "*.db3")))

    if not db3_files:
        print(f"[ERROR] No .db3 files found in: {DATASET_FOLDER}")
        print("Check the DATASET_FOLDER path at the top of this script.")
        return

    print(f"Found {len(db3_files)} recordings. Starting batch latency test...\n")

    all_results = []
    for idx, file_path in enumerate(db3_files, start=1):
        print(f"[{idx}/{len(db3_files)}] Processing: {os.path.basename(file_path)}")
        result = process_single_file(file_path, show_window=False)
        if result is None:
            print(f"   -> No frames could be read from this file, skipping.")
            continue
        all_results.append(result)
        print(f"   -> Avg latency: {result['avg_latency_ms']:.2f} ms | "
              f"Avg FPS: {result['avg_fps']:.2f} | Frames: {result['frames']}")

    if not all_results:
        print("\nNo results collected from any file.")
        return

    # Write summary CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    # Overall stats across ALL videos combined
    overall_avg = np.mean([r["avg_latency_ms"] for r in all_results])
    overall_fps = np.mean([r["avg_fps"] for r in all_results])

    print("\n" + "=" * 50)
    print("BATCH RESULTS SUMMARY")
    print("=" * 50)
    print(f"Files processed:        {len(all_results)}")
    print(f"Overall avg latency:    {overall_avg:.2f} ms")
    print(f"Overall avg FPS:        {overall_fps:.2f}")
    print(f"Per-file results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()