import pyrealsense2 as rs
import time

# 1. Ask for details to automatically name the file
print("--- New Recording Session ---")
exercise = input("Exercise name (e.g., squat, pushup): ").lower()
angle = input("Camera angle (e.g., front, side, 45deg): ").lower()
subject = input("Subject name (e.g., your_name): ").lower()
iteration = input("Set number (e.g., 01): ")

# Create a clean filename
filename = f"{exercise}_{angle}_{subject}_{iteration}.db3"

# 2. Configure the pipeline with OPTIMAL D435i settings
pipeline = rs.pipeline()
config = rs.config()

# Setting to 848x480 at 30 FPS for the best depth quality to processing speed ratio
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

config.enable_record_to_file(filename)

# 3. Start recording
print(f"\n[INFO] Starting camera...")
pipeline.start(config)

# Give the person a second to get into position
time.sleep(2) 
print(f"\n>>> RECORDING: {filename} <<<")
print("Do your reps! Press Ctrl+C in this terminal to stop.")

try:
    while True:
        # Keep capturing frames
        frames = pipeline.wait_for_frames()
        
except KeyboardInterrupt:
    print("\n[INFO] Stopping...")
finally:
    pipeline.stop()
    print(f"[SUCCESS] Data saved as: {filename}")