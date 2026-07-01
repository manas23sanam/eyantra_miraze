import time
import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
import ctypes

from pendulum_detector import PendulumDetector, PendulumSwingTracker

EXERCISE_NAME = "PENDULUM SHOULDER"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

WINDOW_NAME = "Smart Mirror - Pendulum Demo"
INSTRUCTION_DURATION_SEC = 8

INSTRUCTION_LINES = [
    "PENDULUM EXERCISE - How to do it:",
    "1. Bend forward at the hips (torso angle 100-145 degrees)",
    "2. Let one arm hang down vertically",
    "3. Swing the hanging arm back and forth like a pendulum",
    "4. Keep the torso still while swinging the arm",
    "",
    "Press SPACE when you're ready to begin",
]

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


class CustomPendulumSwingTracker(PendulumSwingTracker):
    def __init__(self, window_size=30, min_range_deg=20.0):
        super().__init__(window_size, min_range_deg)
        self.cycle_count = 0
        self.state = "neutral"
        self.prev_angle = None

    def update(self, arm_swing_angle):
        status = super().update(arm_swing_angle)
        is_swinging = status["is_swinging"]

        if arm_swing_angle is not None and self.prev_angle is not None:
            if len(self.history) >= 10:
                midpoint = sum(self.history) / len(self.history)
                if is_swinging:
                    if self.prev_angle < midpoint <= arm_swing_angle:
                        if self.state == "backward":
                            self.cycle_count += 1
                        self.state = "forward"
                    elif self.prev_angle > midpoint >= arm_swing_angle:
                        self.state = "backward"

        if arm_swing_angle is not None:
            self.prev_angle = arm_swing_angle

        status["cycle_count"] = self.cycle_count
        return status


def get_screen_resolution():
    try:
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return 1366, 768


def color_for_state(is_correct):
    return (0, 200, 0) if is_correct else (0, 0, 230)


def draw_skeleton_with_feedback(image, landmarks_proto, is_correct):
    color = color_for_state(is_correct)
    landmark_style = mp_drawing.DrawingSpec(color=color, thickness=4, circle_radius=4)
    connection_style = mp_drawing.DrawingSpec(color=color, thickness=3)
    mp_drawing.draw_landmarks(
        image, landmarks_proto, mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=landmark_style,
        connection_drawing_spec=connection_style,
    )


def draw_instruction_screen(image, seconds_remaining):
    """Full-frame instruction overlay shown before live feedback starts."""
    h, w = image.shape[:2]
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

    y = int(h * 0.28)
    for i, line in enumerate(INSTRUCTION_LINES):
        scale = 1.1 if i == 0 else 0.8
        thickness = 3 if i == 0 else 2
        color = (255, 255, 255) if i == 0 else (210, 210, 210)
        cv2.putText(image, line, (int(w * 0.08), y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)
        y += 48 if i == 0 else 38

    cv2.putText(image, f"Starting in {int(seconds_remaining)}s...", (int(w * 0.08), h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)


def draw_hud(image, exercise_name, torso_bend_angle, swing_range, swing_status, swing_cycles, is_correct, latency_ms, fps):
    h, w = image.shape[:2]

    # Top banner: exercise name + rep count
    cv2.rectangle(image, (0, 0), (w, 60), (40, 40, 40), -1)
    cv2.putText(image, exercise_name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)

    rep_text = f"Swing Cycles: {swing_cycles}"
    rep_size = cv2.getTextSize(rep_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    cv2.putText(image, rep_text, (w - rep_size[0] - 280, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # Correctness cue, top right
    status_color = color_for_state(is_correct)
    cue_text = "GOOD FORM" if is_correct else "IMPROVE POSTURE/SWING"
    cv2.putText(image, cue_text, (w - 260, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, status_color, 2, cv2.LINE_AA)

    # Bottom-left: torso bend and swing details
    if torso_bend_angle is not None:
        cv2.putText(image, f"Torso Bend: {torso_bend_angle:.1f} deg (Target: 100-145)", (20, h - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        status_str = "SWINGING" if swing_status else "STILL"
        cv2.putText(image, f"Swing Range: {swing_range:.1f} deg | Status: {status_str}", (20, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(image, "No pose detected", (20, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

    # Bottom-left: latency / fps
    cv2.putText(image, f"Latency: {latency_ms:.1f} ms   FPS: {fps:.1f}", (20, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


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

    detector = PendulumDetector(depth_intrinsics=intrinsics)
    tracker = CustomPendulumSwingTracker(min_range_deg=20.0)

    latency_window = []
    WINDOW_SIZE = 30

    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print(f"Starting live demo: {EXERCISE_NAME}")
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

            key = cv2.waitKey(1) & 0xFF
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

            # ===================== LATENCY TIMER =====================
            start_time = time.perf_counter()
            result = detector.process_frame(color_image, depth_frame, depth_image, swinging_side="auto")
            end_time = time.perf_counter()
            # ===========================================================

            latency_ms = (end_time - start_time) * 1000
            latency_window.append(latency_ms)
            if len(latency_window) > WINDOW_SIZE:
                latency_window.pop(0)
            avg_latency = sum(latency_window) / len(latency_window)
            live_fps = 1000 / avg_latency if avg_latency > 0 else 0

            torso_bend = None
            arm_swing = None
            swing_range = 0.0
            is_swinging = False
            swing_cycles = 0
            is_correct = False

            if result is not None:
                torso_bend = result["torso_bend_angle_deg"]
                arm_swing = result["arm_swing_angle_deg"]

                status = tracker.update(arm_swing)
                swing_range = status["range_deg"]
                is_swinging = status["is_swinging"]
                swing_cycles = status["cycle_count"]

                is_correct = (torso_bend is not None and 100.0 <= torso_bend <= 145.0 and is_swinging)

                draw_skeleton_with_feedback(display_image, result["landmarks"], is_correct)
            else:
                swing_cycles = tracker.cycle_count

            draw_hud(display_image, EXERCISE_NAME, torso_bend, swing_range, is_swinging,
                     swing_cycles, is_correct, avg_latency, live_fps)

            cv2.imshow(WINDOW_NAME, display_image)

    finally:
        pipeline.stop()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
