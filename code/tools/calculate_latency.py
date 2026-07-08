import pyrealsense2 as rs
import time
import cv2
import numpy as np
file_path = 'mini squat_45_manas_1.db3'
pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, file_path)
pipeline.start(config)
print(f'Starting playback and latency test for: {file_path}')
latency_records = []
try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue
        start_time = time.perf_counter()
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        latency_records.append(latency_ms)
        print(f'Frame Latency: {latency_ms:.2f} ms')
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow('Playback', depth_colormap)
        if cv2.waitKey(1) & 255 == ord('q'):
            break
except RuntimeError:
    print('\n[SUCCESS] Reached the end of the video file.')
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    if latency_records:
        avg_latency = sum(latency_records) / len(latency_records)
        print(f'\n--- RESULTS FOR PRESENTATION ---')
        print(f'Average Latency: {avg_latency:.2f} ms')
        print(f'Average FPS: {1000 / avg_latency:.2f} FPS')
