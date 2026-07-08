import os
import glob
import time
import csv
import numpy as np
import cv2
import pyrealsense2 as rs
from detectors.squat_angle_detector import SquatAngleDetector
DATASET_FOLDER = 'C:\\Users\\BIT\\OneDrive\\Desktop\\software mirror\\new'
OUTPUT_CSV = 'squat_angle_validation_results.csv'

def categorize_filename(filename):
    name = filename.lower()
    if 'wrong' in name:
        if '45' in name:
            return 'wrong_45'
        elif 'front' in name:
            return 'wrong_front'
        elif 'side' in name:
            return 'wrong_side'
        else:
            return 'wrong_other'
    return 'correct'

def process_file(file_path):
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, file_path, repeat_playback=False)
    started = False
    angles = []
    latencies = []
    frame_count = 0
    try:
        profile = pipeline.start(config)
        started = True
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)
        depth_stream = profile.get_stream(rs.stream.depth)
        intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
        detector = SquatAngleDetector(depth_intrinsics=intrinsics)
        align = rs.align(rs.stream.color)
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError:
                break
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            frame_count += 1
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            start_time = time.perf_counter()
            result = detector.process_frame(color_image, depth_frame, depth_image, side='auto')
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)
            if result is not None:
                angles.append(result['knee_angle_deg'])
        detector.close()
    except RuntimeError as e:
        print(f'   [SKIPPED] {e}')
        return None
    finally:
        if started:
            pipeline.stop()
    if not angles:
        print(f'   [WARNING] No pose detected in any frame of this file.')
        return None
    angles_arr = np.array(angles)
    lat_arr = np.array(latencies)
    return {'file': os.path.basename(file_path), 'category': categorize_filename(os.path.basename(file_path)), 'frames_total': frame_count, 'frames_with_pose': len(angles), 'min_knee_angle': float(np.min(angles_arr)), 'max_knee_angle': float(np.max(angles_arr)), 'avg_knee_angle': float(np.mean(angles_arr)), 'avg_latency_ms': float(np.mean(lat_arr)), 'p95_latency_ms': float(np.percentile(lat_arr, 95)), 'avg_fps': float(1000 / np.mean(lat_arr)) if np.mean(lat_arr) > 0 else 0}

def main():
    db3_files = sorted(glob.glob(os.path.join(DATASET_FOLDER, '*.db3')))
    if not db3_files:
        print(f'[ERROR] No .db3 files found in: {DATASET_FOLDER}')
        return
    print(f'Found {len(db3_files)} files. Running squat angle validation...\n')
    results = []
    for idx, file_path in enumerate(db3_files, start=1):
        print(f'[{idx}/{len(db3_files)}] {os.path.basename(file_path)}')
        r = process_file(file_path)
        if r:
            results.append(r)
            print(f"   category={r['category']} | min_angle={r['min_knee_angle']:.1f} max_angle={r['max_knee_angle']:.1f} | avg_latency={r['avg_latency_ms']:.2f}ms")
    if not results:
        print('No results collected.')
        return
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print('\n' + '=' * 60)
    print('SUMMARY BY CATEGORY (use this to pick your angle threshold)')
    print('=' * 60)
    categories = sorted(set((r['category'] for r in results)))
    for cat in categories:
        cat_results = [r for r in results if r['category'] == cat]
        min_angles = [r['min_knee_angle'] for r in cat_results]
        print(f'\n{cat}  ({len(cat_results)} files)')
        print(f'   min knee angle reached -> avg: {np.mean(min_angles):.1f} deg, range: {np.min(min_angles):.1f}-{np.max(min_angles):.1f} deg')
    overall_avg_latency = np.mean([r['avg_latency_ms'] for r in results])
    print(f'\nOverall avg pipeline latency: {overall_avg_latency:.2f} ms ({1000 / overall_avg_latency:.1f} FPS)')
    print(f'Full results saved to: {OUTPUT_CSV}')
if __name__ == '__main__':
    main()
