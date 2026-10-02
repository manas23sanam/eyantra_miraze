import time
import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
import ctypes
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code')))
from shoulder_circle_detector import ShoulderCircleDetector, ShoulderCircleTracker
EXERCISE_NAME = 'SHOULDER CIRCLE'
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
WINDOW_NAME = 'Smart Mirror - Shoulder Circle Demo'
INSTRUCTION_DURATION_SEC = 8
INSTRUCTION_LINES = ['SHOULDER CIRCLE - How to do it:', '1. Stand straight with your arm hanging down', '2. Roll your shoulder/elbow in a smooth circle', '3. Keep the movement consistent in one direction', '4. Complete a full 360-degree rotation', '', "Press SPACE when you're ready to begin"]
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def get_screen_resolution():
    try:
        user32 = ctypes.windll.user32
        return (user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))
    except Exception:
        return (1366, 768)

def color_for_state(is_correct):
    return (0, 200, 0) if is_correct else (0, 0, 230)

def draw_skeleton_with_feedback(image, landmarks_proto, is_correct):
    color = color_for_state(is_correct)
    landmark_style = mp_drawing.DrawingSpec(color=color, thickness=4, circle_radius=4)
    connection_style = mp_drawing.DrawingSpec(color=color, thickness=3)
    mp_drawing.draw_landmarks(image, landmarks_proto, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=landmark_style, connection_drawing_spec=connection_style)

def draw_instruction_screen(image, seconds_remaining):
    h, w = image.shape[:2]
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)
    y = int(h * 0.28)
    for i, line in enumerate(INSTRUCTION_LINES):
        scale = 1.1 if i == 0 else 0.8
        thickness = 3 if i == 0 else 2
        color = (255, 255, 255) if i == 0 else (210, 210, 210)
        cv2.putText(image, line, (int(w * 0.08), y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        y += 48 if i == 0 else 38
    cv2.putText(image, f'Starting in {int(seconds_remaining)}s...', (int(w * 0.08), h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

def draw_hud(image, exercise_name, rotation, reversal_count, direction, correct_reps, is_correct, latency_ms, fps):
    h, w = image.shape[:2]
    cv2.rectangle(image, (0, 0), (w, 60), (40, 40, 40), -1)
    cv2.putText(image, exercise_name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    rep_text = f'Correct Reps: {correct_reps}'
    rep_size = cv2.getTextSize(rep_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    cv2.putText(image, rep_text, (w - rep_size[0] - 280, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    status_color = color_for_state(is_correct)
    cue_text = 'GOOD FORM' if is_correct else 'TOO MANY REVERSALS'
    cv2.putText(image, cue_text, (w - 260, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)
    if rotation is not None:
        dir_str = direction.upper() if direction else 'NONE'
        cv2.putText(image, f'Rotation: {rotation:.1f} deg | Dir: {dir_str}', (20, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f'Reversals: {reversal_count}', (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(image, 'No pose detected', (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f'Latency: {latency_ms:.1f} ms   FPS: {fps:.1f}', (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

def main():
    screen_w, screen_h = get_screen_resolution()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_stream = profile.get_stream(rs.stream.depth)
    intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
    detector = ShoulderCircleDetector(depth_intrinsics=intrinsics)
    tracker = ShoulderCircleTracker()
    latency_window = []
    WINDOW_SIZE = 30
    correct_reps = 0
    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    print(f'Starting live demo: {EXERCISE_NAME}')
    print("Press 'q' to quit, SPACE to skip instructions.")
    instruction_start_time = time.perf_counter()
    in_instruction_phase = True
    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            display_image = cv2.resize(color_image, (screen_w, screen_h))
            key = cv2.waitKey(1) & 255
            if key == ord('q'):
                break
            if in_instruction_phase:
                elapsed = time.perf_counter() - instruction_start_time
                remaining = max(0, INSTRUCTION_DURATION_SEC - elapsed)
                draw_instruction_screen(display_image, remaining)
                if remaining <= 0 or key == ord(' '):
                    in_instruction_phase = False
                cv2.imshow(WINDOW_NAME, display_image)
                continue
            start_time = time.perf_counter()
            angle, landmarks = detector.get_shoulder_elbow_angle(color_image, depth_frame, depth_image, side='auto')
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latency_window.append(latency_ms)
            if len(latency_window) > WINDOW_SIZE:
                latency_window.pop(0)
            avg_latency = sum(latency_window) / len(latency_window)
            live_fps = 1000 / avg_latency if avg_latency > 0 else 0
            rotation = None
            reversal_count = 0
            direction = None
            is_correct = True
            if landmarks is not None:
                status = tracker.update(angle)
                rotation = status['accumulated_rotation_deg']
                reversal_count = status['reversal_count']
                direction = status['direction']
                is_complete = status['is_complete']
                is_correct = reversal_count <= 1
                if is_complete:
                    if is_correct:
                        correct_reps += 1
                    tracker.reset()
                draw_skeleton_with_feedback(display_image, landmarks, is_correct)
            draw_hud(display_image, EXERCISE_NAME, rotation, reversal_count, direction, correct_reps, is_correct, avg_latency, live_fps)
            cv2.imshow(WINDOW_NAME, display_image)
    finally:
        pipeline.stop()
        detector.close()
        cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
