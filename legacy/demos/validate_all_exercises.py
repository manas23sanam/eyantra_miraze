import os
import glob
import numpy as np
import cv2
import pyrealsense2 as rs
from detectors.sit_to_stand_detector import SitToStandDetector
from detectors.shoulder_circle_detector import ShoulderCircleDetector, ShoulderCircleTracker
from detectors.pendulum_detector import PendulumDetector, PendulumSwingTracker
from detectors.upper_trapezius_detector import UpperTrapeziusDetector
EXERCISE_FOLDERS = {'sit_to_stand': 'C:\\path\\to\\sit_to_stand_samples', 'shoulder_circle': 'C:\\path\\to\\shoulder_circle_samples', 'pendulum': 'C:\\path\\to\\pendulum_samples', 'upper_trapezius': 'C:\\path\\to\\upper_trapezius_samples'}

def open_db3(file_path):
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, file_path, repeat_playback=False)
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
    align = rs.align(rs.stream.color)
    depth_stream = profile.get_stream(rs.stream.depth)
    intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
    return (pipeline, align, intrinsics)

def validate_sit_to_stand(folder):
    files = sorted(glob.glob(os.path.join(folder, '*.db3')))
    detector_intrinsics_cache = None
    print(f'\n--- SIT TO STAND ({len(files)} files) ---')
    for fp in files:
        pipeline, align, intrinsics = open_db3(fp)
        detector = SitToStandDetector(depth_intrinsics=intrinsics)
        angles = []
        try:
            while True:
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=2000)
                except RuntimeError:
                    break
                frames = align.process(frames)
                depth_frame, color_frame = (frames.get_depth_frame(), frames.get_color_frame())
                if not depth_frame or not color_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                result = detector.process_frame(color_image, depth_frame, depth_image)
                if result:
                    angles.append(result['hip_knee_ankle_angle_deg'])
        finally:
            pipeline.stop()
            detector.close()
        if angles:
            print(f'  {os.path.basename(fp)}: min={min(angles):.1f} max={max(angles):.1f} (expect near-180 standing, ~90-110 sitting)')
        else:
            print(f'  {os.path.basename(fp)}: NO POSE DETECTED - check framing/lighting')

def validate_upper_trapezius(folder):
    files = sorted(glob.glob(os.path.join(folder, '*.db3')))
    print(f'\n--- UPPER TRAPEZIUS ({len(files)} files) ---')
    for fp in files:
        pipeline, align, intrinsics = open_db3(fp)
        detector = UpperTrapeziusDetector(depth_intrinsics=intrinsics)
        tilts = []
        try:
            while True:
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=2000)
                except RuntimeError:
                    break
                frames = align.process(frames)
                depth_frame, color_frame = (frames.get_depth_frame(), frames.get_color_frame())
                if not depth_frame or not color_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                result = detector.process_frame(color_image, depth_frame, depth_image)
                if result:
                    tilts.append(result['neck_tilt_deg'])
        finally:
            pipeline.stop()
            detector.close()
        if tilts:
            print(f'  {os.path.basename(fp)}: min={min(tilts):.1f} max={max(tilts):.1f} (neutral near 0, full stretch likely 20-40+ deg)')
        else:
            print(f'  {os.path.basename(fp)}: NO POSE DETECTED')

def validate_shoulder_circle(folder):
    files = sorted(glob.glob(os.path.join(folder, '*.db3')))
    print(f'\n--- SHOULDER CIRCLE ({len(files)} files) ---')
    for fp in files:
        pipeline, align, intrinsics = open_db3(fp)
        detector = ShoulderCircleDetector(depth_intrinsics=intrinsics)
        tracker = ShoulderCircleTracker()
        try:
            while True:
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=2000)
                except RuntimeError:
                    break
                frames = align.process(frames)
                depth_frame, color_frame = (frames.get_depth_frame(), frames.get_color_frame())
                if not depth_frame or not color_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                angle, _ = detector.get_shoulder_elbow_angle(color_image, depth_frame, depth_image)
                tracker.update(angle)
        finally:
            pipeline.stop()
            detector.close()
        status = tracker._status()
        print(f"  {os.path.basename(fp)}: accumulated_rotation={status['accumulated_rotation_deg']:.0f} deg reversals={status['reversal_count']} (full circle ~360)")

def validate_pendulum(folder):
    files = sorted(glob.glob(os.path.join(folder, '*.db3')))
    print(f'\n--- PENDULUM ({len(files)} files) ---')
    for fp in files:
        pipeline, align, intrinsics = open_db3(fp)
        detector = PendulumDetector(depth_intrinsics=intrinsics)
        swing_tracker = PendulumSwingTracker()
        bend_angles = []
        try:
            while True:
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=2000)
                except RuntimeError:
                    break
                frames = align.process(frames)
                depth_frame, color_frame = (frames.get_depth_frame(), frames.get_color_frame())
                if not depth_frame or not color_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                result = detector.process_frame(color_image, depth_frame, depth_image)
                if result:
                    bend_angles.append(result['torso_bend_angle_deg'])
                    swing_tracker.update(result['arm_swing_angle_deg'])
        finally:
            pipeline.stop()
            detector.close()
        swing_status = swing_tracker.history
        if bend_angles:
            print(f'  {os.path.basename(fp)}: torso_bend min={min(bend_angles):.1f} max={max(bend_angles):.1f} | arm_swing_range={(max(swing_status) - min(swing_status) if swing_status else 0):.1f} deg')
        else:
            print(f'  {os.path.basename(fp)}: NO POSE DETECTED')

def main():
    if os.path.isdir(EXERCISE_FOLDERS['sit_to_stand']):
        validate_sit_to_stand(EXERCISE_FOLDERS['sit_to_stand'])
    if os.path.isdir(EXERCISE_FOLDERS['upper_trapezius']):
        validate_upper_trapezius(EXERCISE_FOLDERS['upper_trapezius'])
    if os.path.isdir(EXERCISE_FOLDERS['shoulder_circle']):
        validate_shoulder_circle(EXERCISE_FOLDERS['shoulder_circle'])
    if os.path.isdir(EXERCISE_FOLDERS['pendulum']):
        validate_pendulum(EXERCISE_FOLDERS['pendulum'])
if __name__ == '__main__':
    main()
