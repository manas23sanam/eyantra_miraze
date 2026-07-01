import time
import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
import ctypes

from squat_angle_detector import SquatAngleDetector


EXERCISE_NAME = "MINI SQUAT"
SQUAT_ANGLE_MAX = 170.0
SQUAT_ANGLE_MIN = 130.0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

LANDMARK_VISIBILITY_FLOOR = 0.15

WINDOW_NAME = "Smart Mirror - Live Demo"

# How long to show the instruction screen before switching to live feedback,
# in seconds. The person can also press SPACE to skip ahead once they're ready.
INSTRUCTION_DURATION_SEC = 8

INSTRUCTION_LINES = [
    "MINI SQUAT - How to do it:",
    "1. Stand with feet shoulder-width apart",
    "2. Bend your knees slightly, like sitting back into a chair",
    "3. Keep your back straight, don't lean forward",
    "4. Go down a little, then stand back up - that's one rep",
    "",
    "Press SPACE when you're ready to begin",
]

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def get_screen_resolution():
    """
    Auto-detects the actual screen resolution in pixels so the demo window
    fills the laptop screen correctly regardless of physical screen size or
    DPI scaling. Falls back to a common resolution if detection fails (e.g.
    on a non-Windows OS where this method isn't available).
    """
    try:
        user32 = ctypes.windll.user32
        width = user32.GetSystemMetrics(0)
        height = user32.GetSystemMetrics(1)
        return width, height
    except Exception:
        return 1366, 768  # safe fallback


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


def draw_hud(image, exercise_name, knee_angle, is_correct, latency_ms, fps, cue_text, rep_count):
    h, w = image.shape[:2]

    # Top banner: exercise name + rep count
    cv2.rectangle(image, (0, 0), (w, 60), (40, 40, 40), -1)
    cv2.putText(image, exercise_name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)

    rep_text = f"Reps: {rep_count}"
    rep_size = cv2.getTextSize(rep_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    cv2.putText(image, rep_text, (w - rep_size[0] - 280, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # Correctness cue, top right
    status_color = color_for_state(is_correct)
    cv2.putText(image, cue_text, (w - 260, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, status_color, 2, cv2.LINE_AA)

    # Bottom-left: angle reading
    if knee_angle is not None:
        cv2.putText(image, f"Knee angle: {knee_angle:.1f} deg", (20, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(image, "No pose detected", (20, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

    # Bottom-left: latency / fps
    cv2.putText(image, f"Latency: {latency_ms:.1f} ms   FPS: {fps:.1f}", (20, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


class RepCounter:
    """
    Tracks squat reps using the same MIN/MAX angle band as the correctness
    check. A rep = going from standing (angle above MAX) down into the
    correct squat zone (between MIN and MAX) and back up to standing.
    Deliberately does NOT count a rep if the person goes too deep
    (below MIN) without ever passing through the correct zone on the way
    back up - that would be a malformed rep, not a good one.
    """
    def __init__(self, stand_angle=170.0, squat_zone=(130.0, 170.0)):
        self.stand_angle = stand_angle
        self.squat_min, self.squat_max = squat_zone
        self.state = "standing"
        self.rep_count = 0
        self.reached_correct_zone_this_rep = False

    def update(self, angle):
        if angle is None:
            return self.rep_count, False

        just_completed = False

        if self.state == "standing":
            if angle <= self.squat_max:
                self.state = "descending"
                self.reached_correct_zone_this_rep = False

        if self.state in ("descending", "bottom"):
            if self.squat_min <= angle <= self.squat_max:
                self.reached_correct_zone_this_rep = True
                self.state = "bottom"

        if self.state in ("descending", "bottom") and angle >= self.stand_angle:
            # back to standing - completed a rep cycle
            if self.reached_correct_zone_this_rep:
                self.rep_count += 1
                just_completed = True
            self.state = "standing"

        return self.rep_count, just_completed


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

    detector = SquatAngleDetector(depth_intrinsics=intrinsics)
    rep_counter = RepCounter(stand_angle=SQUAT_ANGLE_MAX, squat_zone=(SQUAT_ANGLE_MIN, SQUAT_ANGLE_MAX))

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

            # Resize the live feed to fill the actual screen resolution
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
            result = detector.process_frame(color_image, depth_frame, depth_image, side="auto")
            end_time = time.perf_counter()
            # ===========================================================

            latency_ms = (end_time - start_time) * 1000
            latency_window.append(latency_ms)
            if len(latency_window) > WINDOW_SIZE:
                latency_window.pop(0)
            avg_latency = sum(latency_window) / len(latency_window)
            live_fps = 1000 / avg_latency if avg_latency > 0 else 0

            knee_angle = None
            is_correct = False
            cue_text = "Step into frame"

            if result is not None:
                knee_angle = result["knee_angle_deg"]
                is_correct = SQUAT_ANGLE_MIN <= knee_angle <= SQUAT_ANGLE_MAX

                if knee_angle > SQUAT_ANGLE_MAX:
                    cue_text = "GO LOWER"
                elif knee_angle < SQUAT_ANGLE_MIN:
                    cue_text = "TOO DEEP - EASE UP"
                else:
                    cue_text = "GOOD DEPTH"

                rep_count, just_completed = rep_counter.update(knee_angle)

                # Scale landmark coordinates are normalized (0-1), so drawing
                # works fine directly on the resized display_image - MediaPipe
                # landmarks are resolution-independent.
                draw_skeleton_with_feedback(display_image, result["landmarks"], is_correct)
            else:
                rep_count = rep_counter.rep_count

            draw_hud(display_image, EXERCISE_NAME, knee_angle, is_correct,
                     avg_latency, live_fps, cue_text, rep_count)

            cv2.imshow(WINDOW_NAME, display_image)

    finally:
        pipeline.stop()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()