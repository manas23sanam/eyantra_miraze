import os
import glob
import pandas as pd
import pyrealsense2 as rs
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

# Find all .db3 files in the current folder automatically
video_files = glob.glob("*.db3")
all_data = []

print(f"Found {len(video_files)} files in this folder to process...")

for file_path in video_files:
    filename = os.path.basename(file_path)
    print(f"Processing: {filename}...")
    
    # Extract labels from the filename format (e.g., squat_front_subject_01.db3)
    parts = filename.replace(".db3", "").split("_")
    exercise = parts[0] if len(parts) > 0 else "unknown"
    angle = parts[1] if len(parts) > 1 else "unknown"
    
    # Set up RealSense playback from file
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(file_path, repeat_playback=False)
    try:
        profile = pipeline.start(config)
        playback = profile.get_device().as_playback()
        playback.set_real_time(False) # Force processing every frame, don't skip
        align = rs.align(rs.stream.color)
        
        frame_count = 0
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
                
            intrinsics = color_frame.get_profile().as_video_stream_profile().get_intrinsics()
            color_image = np.asanyarray(color_frame.get_data())
            h, w, _ = color_image.shape
            
            # Run MediaPipe tracking
            results = pose.process(color_image)
            
            if results.pose_landmarks:
                frame_count += 1
                
                
                # --- ADD THESE TWO LINES ---
                if frame_count % 30 == 0:
                    print(f"  -> Extracted {frame_count} frames...")
                
                # Start a new row for the CSV
                row = {
                    "video_file": filename,
                    "exercise": exercise,
                    "camera_angle": angle,
                    "frame_number": frame_count
                }
                
                # Extract all 33 body joints in 3D
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    joint_name = mp_pose.PoseLandmark(idx).name.lower()
                    pixel_x = int(landmark.x * w)
                    pixel_y = int(landmark.y * h)
                    
                    x, y, z = 0.0, 0.0, 0.0
                    if 0 <= pixel_x < w and 0 <= pixel_y < h:
                        depth = depth_frame.get_distance(pixel_x, pixel_y)
                        if depth > 0:
                            # Mathematical deprojection to real-world meters
                            point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [pixel_x, pixel_y], depth)
                            x, y, z = point_3d[0], point_3d[1], point_3d[2]
                    
                    # Add X, Y, Z columns for this specific joint to the row
                    row[f"{joint_name}_x"] = round(x, 4)
                    row[f"{joint_name}_y"] = round(y, 4)
                    row[f"{joint_name}_z"] = round(z, 4)
                    
                all_data.append(row)
                
    except RuntimeError:
        # File finished playing naturally
        print(f"Finished parsing {filename}. Extracted {frame_count} valid frames.")
    finally:
        pipeline.stop()

# Convert everything to a Pandas DataFrame and save to CSV
df = pd.DataFrame(all_data)
output_filename = "exercise_3d_dataset.csv"
df.to_csv(output_filename, index=False)

print(f"\n[SUCCESS] Master dataset successfully saved as '{output_filename}'!")
print(f"Total frames extracted across all videos: {len(df)}")